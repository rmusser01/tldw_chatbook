"""Pure display-state contracts for the Library prompts canvas."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
import sqlite3
from datetime import datetime, timezone

import pytest

import tldw_chatbook.Library.library_prompts_state as prompts_state_module
from tldw_chatbook.DB.Prompts_DB import ConflictError
from tldw_chatbook.Library.library_prompts_state import (
    PromptArtifactDraft,
    PromptHistoryRestoreOutcome,
    PromptListRow,
    PromptSelectionBasket,
    PromptSelectionEntry,
    apply_prompt_history_count,
    apply_prompt_history_page,
    apply_prompt_history_preview,
    apply_prompt_history_restore,
    begin_prompt_history_count,
    begin_prompt_history_page,
    begin_prompt_history_preview,
    begin_prompt_history_restore,
    build_prompt_history_page,
    build_prompt_history_state,
    close_prompt_history,
    definition_state_display_label,
    prepare_prompt_artifact_save,
    build_prompt_editor_state,
    build_prompts_list_state,
    classify_prompt_save_error,
    coerce_prompt_editor_mode,
    format_prompt_history_restore_outcome,
    history_restore_gate,
    prompt_history_count_label,
    prompt_editor_meta_line,
    prompt_basic_unavailable_reason,
    reset_prompt_history_page,
    require_artifact_save_supported,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    blank_recipe,
    outcome_first_recipe,
)
from tldw_chatbook.Prompt_Management.prompt_normalizers import (
    normalize_prompt_history_page,
    normalize_prompt_list,
)
from tldw_chatbook.Prompt_Management.prompt_restore_errors import (
    PromptRestoreError,
    PromptRestoreErrorCode,
)
from tldw_chatbook.Prompt_Management.prompt_source_capabilities import (
    PromptCapabilityError,
    PromptSourceCapabilities,
    local_prompt_capabilities,
)

NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)


class _PromptSelectionInt(int):
    """An integer lookalike that strict selection boundaries must reject."""


class _PromptSelectionStr(str):
    """A text lookalike that strict selection boundaries must reject."""


class _PromptSelectionTuple(tuple):
    """A tuple lookalike that strict selection boundaries must reject."""


PROMPT_A = {
    "id": 1,
    "name": "Summarize",
    "author": "Alice",
    "details": "Summarizes text",
    "system_prompt": "You are helpful.",
    "user_prompt": "Summarize: {text}",
    "keywords": ["writing", "summary"],
    "last_modified": "2026-07-07T11:57:00+00:00",
    "version": 2,
}
PROMPT_B = {
    "id": 2,
    "name": "brainstorm",
    "author": "",
    "keywords": [],
    "last_modified": "2026-07-06T12:00:00+00:00",
    "version": 1,
}
PROMPT_C = {
    "id": 3,
    "name": "Zeta ideas",
    "author": None,
    "details": "Ideas for the offsite",
    "keywords": ["kw1", "kw2"],
    "last_modified": "2026-07-07T11:00:00+00:00",
    "version": 1,
}


def _browse_prompt(record):
    local_id = record["id"]
    return {**record, "id": f"local:prompt:{local_id}", "local_id": local_id}


BROWSE_PROMPT_A = _browse_prompt(PROMPT_A)
BROWSE_PROMPT_B = _browse_prompt(PROMPT_B)
BROWSE_PROMPT_C = _browse_prompt(PROMPT_C)


def test_browse_prompt_scope_defaults_to_library_twenty_row_pages():
    scope = prompts_state_module.PromptBrowseScope()

    assert prompts_state_module.DEFAULT_PROMPT_BROWSE_PAGE_SIZE == 20
    assert scope.page_size == 20


def test_browse_prompt_scope_normalizes_query_sort_and_bounded_page_size():
    scope = prompts_state_module.PromptBrowseScope(
        query="  alpha beta \n",
        collection_id=7,
        sort_by=" NAME ",
        sort_order=" ASC ",
        page=2,
        page_size=prompts_state_module.MAX_PROMPT_BROWSE_PAGE_SIZE + 500,
    )

    assert scope.backend == "local"
    assert scope.query == "alpha beta"
    assert scope.collection_id == 7
    assert scope.sort_by == "name"
    assert scope.sort_order == "asc"
    assert scope.page == 2
    assert scope.page_size == prompts_state_module.MAX_PROMPT_BROWSE_PAGE_SIZE
    assert (
        scope.fingerprint
        == prompts_state_module.PromptBrowseScope(
            query="alpha beta",
            collection_id=7,
            sort_by="name",
            sort_order="asc",
            page=2,
            page_size=prompts_state_module.MAX_PROMPT_BROWSE_PAGE_SIZE,
        ).fingerprint
    )
    assert replace(scope, page=3).fingerprint != scope.fingerprint
    with pytest.raises(FrozenInstanceError):
        scope.query = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "server"}, "local"),
        ({"collection_id": 0}, "collection_id"),
        ({"collection_id": True}, "collection_id"),
        ({"sort_by": "name; DROP TABLE Prompts"}, "sort_by"),
        ({"sort_order": "sideways"}, "sort_order"),
        ({"page": 0}, "page"),
        ({"page": True}, "page"),
        ({"page_size": 0}, "page_size"),
        ({"page_size": True}, "page_size"),
        ({"query": None}, "query"),
    ],
)
def test_browse_prompt_scope_rejects_invalid_public_inputs(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        prompts_state_module.PromptBrowseScope(**kwargs)


@pytest.mark.parametrize(
    "changes",
    [
        {"query": "changed"},
        {"collection_id": 7},
        {"sort_by": "name"},
        {"sort_order": "asc"},
        {"page": 2},
        {"page_size": 51},
    ],
)
def test_browse_prompt_scope_fingerprint_covers_every_variable_field(changes):
    scope = prompts_state_module.PromptBrowseScope()

    assert replace(scope, **changes).fingerprint != scope.fingerprint


def test_prompt_collection_catalog_appends_complete_bounded_pages_literally():
    loading = prompts_state_module.begin_prompt_collection_catalog(
        query="  [bold]  ", request_token=7
    )

    first = prompts_state_module.apply_prompt_collection_catalog_page(
        loading,
        {
            "collections": [
                {
                    "collection_id": collection_id,
                    "name": f"[bold] {collection_id}",
                    "display_name": (
                        "[bold] · #1"
                        if collection_id == 1
                        else f"[bold] {collection_id}"
                    ),
                    "prompt_ids": [],
                    "backend": "local",
                }
                for collection_id in range(1, 101)
            ],
            "limit": 100,
            "offset": 0,
            "total": 207,
        },
        request_token=7,
    )
    second = prompts_state_module.apply_prompt_collection_catalog_page(
        first,
        {
            "collections": [
                {
                    "collection_id": collection_id,
                    "name": f"集合 {collection_id}",
                    "display_name": f"集合 {collection_id}",
                    "prompt_ids": [],
                    "backend": "local",
                }
                for collection_id in range(101, 201)
            ],
            "limit": 100,
            "offset": 100,
            "total": 207,
        },
        request_token=7,
        append=True,
    )

    assert loading.query == "[bold]"
    assert second.status == "ready"
    assert second.total == 207
    assert len(second.items) == 200
    assert second.has_more is True
    assert second.next_offset == 200
    assert second.items[0].display_name == "[bold] · #1"
    assert second.items[-1].display_name == "集合 200"


def test_prompt_collection_catalog_rejects_late_page_and_cross_query_append():
    current = prompts_state_module.begin_prompt_collection_catalog(
        query="current", request_token=9
    )
    stale_record = {
        "collections": [],
        "limit": 100,
        "offset": 0,
        "total": 0,
    }

    assert (
        prompts_state_module.apply_prompt_collection_catalog_page(
            current, stale_record, request_token=8
        )
        is current
    )
    with pytest.raises(ValueError, match="offset"):
        prompts_state_module.apply_prompt_collection_catalog_page(
            current,
            {**stale_record, "offset": 100},
            request_token=9,
            append=True,
        )


def test_prompt_collection_catalog_rejects_duplicate_ids_and_total_drift():
    loading = prompts_state_module.begin_prompt_collection_catalog(
        query="", request_token=11
    )
    first = prompts_state_module.apply_prompt_collection_catalog_page(
        loading,
        {
            "collections": [
                {
                    "collection_id": 1,
                    "name": "One",
                    "display_name": "One",
                    "backend": "local",
                }
            ],
            "limit": 100,
            "offset": 0,
            "total": 2,
        },
        request_token=11,
    )
    for next_page in (
        {
            "collections": [
                {
                    "collection_id": 1,
                    "name": "Again",
                    "display_name": "Again",
                    "backend": "local",
                }
            ],
            "limit": 100,
            "offset": 1,
            "total": 2,
        },
        {
            "collections": [
                {
                    "collection_id": 2,
                    "name": "Two",
                    "display_name": "Two",
                    "backend": "local",
                }
            ],
            "limit": 100,
            "offset": 1,
            "total": 3,
        },
    ):
        with pytest.raises(ValueError):
            prompts_state_module.apply_prompt_collection_catalog_page(
                first,
                next_page,
                request_token=11,
                append=True,
            )


def test_prompt_memberships_stage_and_apply_without_content_save_coupling():
    loading = prompts_state_module.begin_prompt_memberships(
        prompt_id=41,
        identity_fingerprint="local:prompt:41:v3",
        request_token=3,
    )
    ready = prompts_state_module.apply_prompt_memberships_loaded(
        loading,
        collection_ids=(2, 7),
        labels={2: "[bold] literal", 7: "研究"},
        request_token=3,
    )
    staged = prompts_state_module.stage_prompt_memberships(ready, (7, 9))
    applying = prompts_state_module.begin_prompt_memberships_apply(
        staged, request_token=4
    )
    applied = prompts_state_module.apply_prompt_memberships_saved(
        applying,
        collection_ids=(7, 9),
        request_token=4,
    )

    assert not hasattr(ready, "summary")
    assert staged.applied_ids == (2, 7)
    assert staged.staged_ids == (7, 9)
    assert staged.can_apply is True
    assert applying.status == "applying"
    assert applied.applied_ids == (7, 9)
    assert applied.staged_ids == (7, 9)
    assert applied.labels == ((7, "研究"),)
    assert applied.status == "success"
    assert applied.outcome == "Memberships applied."
    assert not hasattr(applied, "content_dirty")
    assert not hasattr(applied, "save_status")


def test_prompt_memberships_reject_stale_apply_and_disable_unsaved_identity():
    disabled = prompts_state_module.disable_prompt_memberships(
        "Save this prompt before managing collections."
    )
    loading = prompts_state_module.begin_prompt_memberships(
        prompt_id=41,
        identity_fingerprint="local:prompt:41:v3",
        request_token=5,
    )
    ready = prompts_state_module.apply_prompt_memberships_loaded(
        loading,
        collection_ids=(2,),
        labels={2: "Work"},
        request_token=5,
    )
    applying = prompts_state_module.begin_prompt_memberships_apply(
        prompts_state_module.stage_prompt_memberships(ready, ()), request_token=6
    )

    assert disabled.can_manage is False
    assert disabled.disabled_reason == "Save this prompt before managing collections."
    assert disabled.can_apply is False
    assert (
        prompts_state_module.apply_prompt_memberships_saved(
            applying,
            collection_ids=(),
            request_token=5,
        )
        is applying
    )


def test_prompt_membership_load_error_blocks_mutation_but_apply_error_retries():
    loading = prompts_state_module.begin_prompt_memberships(
        prompt_id=41,
        identity_fingerprint="local:prompt:41:v3",
        request_token=7,
    )
    load_error = prompts_state_module.fail_prompt_memberships(
        loading,
        request_token=7,
        error="Couldn't load memberships. Retry.",
        phase="load",
    )

    assert load_error.status == "load_error"
    assert load_error.can_manage is False
    assert load_error.can_retry_load is True
    assert prompts_state_module.stage_prompt_memberships(load_error, (9,)) is load_error
    assert (
        prompts_state_module.begin_prompt_memberships_apply(load_error, request_token=8)
        is load_error
    )

    ready = prompts_state_module.apply_prompt_memberships_loaded(
        loading,
        collection_ids=(2,),
        labels={2: "Current"},
        request_token=7,
    )
    staged = prompts_state_module.stage_prompt_memberships(ready, (2, 9))
    applying = prompts_state_module.begin_prompt_memberships_apply(
        staged, request_token=8
    )
    apply_error = prompts_state_module.fail_prompt_memberships(
        applying,
        request_token=8,
        error="Couldn't apply memberships. Retry.",
        phase="apply",
    )

    assert apply_error.status == "apply_error"
    assert apply_error.applied_ids == (2,)
    assert apply_error.staged_ids == (2, 9)
    assert apply_error.can_manage is True
    assert apply_error.can_retry_load is False
    assert apply_error.can_apply is True


def test_prompt_membership_state_rejects_ambiguous_active_and_label_shapes():
    loading = prompts_state_module.begin_prompt_memberships(
        prompt_id=41,
        identity_fingerprint="local:prompt:41:v3",
        request_token=7,
    )
    ready = prompts_state_module.apply_prompt_memberships_loaded(
        loading,
        collection_ids=(2,),
        labels={2: "Current"},
        request_token=7,
    )
    load_error = prompts_state_module.fail_prompt_memberships(
        loading,
        request_token=7,
        error="Couldn't load memberships. Retry.",
        phase="load",
    )
    disabled = prompts_state_module.disable_prompt_memberships("Save first.")

    invalid_states = (
        (loading, {"identity_fingerprint": ""}),
        (loading, {"identity_fingerprint": "   "}),
        (loading, {"request_token": 0}),
        (disabled, {"applied_ids": (2,)}),
        (disabled, {"labels": ((2, "Current"),)}),
        (disabled, {"outcome": "Not allowed"}),
        (ready, {"labels": ((2, "Current"), (2, "Duplicate"))}),
        (ready, {"labels": ((0, "Invalid"),)}),
        (ready, {"labels": ((9, "Unrelated"),)}),
        (load_error, {"outcome": ""}),
        (load_error, {"outcome": "x" * 201}),
    )
    for state, changes in invalid_states:
        with pytest.raises(ValueError):
            replace(state, **changes)


def test_browse_prompt_result_preserves_exact_total_pages_and_clamped_page():
    scope = prompts_state_module.PromptBrowseScope(page=9, page_size=2)

    result = prompts_state_module.build_prompt_browse_result(
        scope,
        {
            "items": [BROWSE_PROMPT_C],
            "total_items": 5,
            "total_pages": 3,
            "current_page": 3,
            "page": 3,
            "per_page": 2,
        },
    )

    assert isinstance(result, prompts_state_module.PromptBrowseResult)
    assert result.scope.page == 3
    assert result.scope_fingerprint == replace(scope, page=3).fingerprint
    assert result.items[0]["id"] == BROWSE_PROMPT_C["id"]
    assert result.items[0]["keywords"] == tuple(PROMPT_C["keywords"])
    assert result.total_items == 5
    assert result.total_pages == 3
    assert result.page == 3
    assert result.status == "ready"


def test_browse_prompt_result_rejects_divergent_page_alias():
    scope = prompts_state_module.PromptBrowseScope(page=2, page_size=2)

    with pytest.raises(ValueError, match="page.*current_page"):
        prompts_state_module.build_prompt_browse_result(
            scope,
            {
                "items": [BROWSE_PROMPT_A, BROWSE_PROMPT_B],
                "total_items": 5,
                "total_pages": 3,
                "current_page": 2,
                "page": 1,
                "per_page": 2,
            },
        )


@pytest.mark.parametrize("field", ["current_page", "page", "per_page"])
@pytest.mark.parametrize("value", [0, True, 1.5])
def test_browse_prompt_product_path_rejects_malformed_page_metadata(field, value):
    scope = prompts_state_module.PromptBrowseScope(page=2)
    payload = {
        "items": [
            {
                "id": 7,
                "uuid": "prompt-7",
                "name": "Seven",
                "version": 1,
            }
        ],
        "total_items": 21,
        "total_pages": 2,
        "current_page": 2,
        "page": 2,
        "per_page": 20,
    }
    payload[field] = value

    if type(value) is int:
        normalized = normalize_prompt_list(
            payload, backend="local", page=scope.page, per_page=scope.page_size
        )
        assert normalized[field] == 0
        with pytest.raises(ValueError, match=field):
            prompts_state_module.build_prompt_browse_result(scope, normalized)
    else:
        with pytest.raises(TypeError, match=field):
            normalize_prompt_list(
                payload, backend="local", page=scope.page, per_page=scope.page_size
            )


@pytest.mark.parametrize(
    ("scope", "items", "total_items", "total_pages", "current_page"),
    [
        (
            prompts_state_module.PromptBrowseScope(page=2, page_size=2),
            [BROWSE_PROMPT_A, BROWSE_PROMPT_B],
            5,
            3,
            2,
        ),
        (
            prompts_state_module.PromptBrowseScope(page=3, page_size=2),
            [BROWSE_PROMPT_C],
            5,
            3,
            3,
        ),
        (
            prompts_state_module.PromptBrowseScope(page_size=2),
            [],
            0,
            0,
            1,
        ),
    ],
)
def test_browse_prompt_result_requires_exact_page_cardinality(
    scope, items, total_items, total_pages, current_page
):
    result = prompts_state_module.build_prompt_browse_result(
        scope,
        {
            "items": items,
            "total_items": total_items,
            "total_pages": total_pages,
            "current_page": current_page,
            "page": current_page,
            "per_page": scope.page_size,
        },
    )

    assert len(result.items) == len(items)


def test_browse_prompt_result_rejects_overfull_partial_last_page():
    scope = prompts_state_module.PromptBrowseScope(page=3, page_size=2)

    with pytest.raises(ValueError, match="item count"):
        prompts_state_module.build_prompt_browse_result(
            scope,
            {
                "items": [BROWSE_PROMPT_A, BROWSE_PROMPT_B],
                "total_items": 5,
                "total_pages": 3,
                "current_page": 3,
                "page": 3,
                "per_page": 2,
            },
        )


@pytest.mark.parametrize(
    ("scope", "items", "total_items", "total_pages", "current_page"),
    [
        (
            prompts_state_module.PromptBrowseScope(page=2, page_size=2),
            [BROWSE_PROMPT_A],
            5,
            3,
            2,
        ),
        (
            prompts_state_module.PromptBrowseScope(page=3, page_size=2),
            [],
            5,
            3,
            3,
        ),
    ],
)
def test_browse_prompt_result_rejects_underfilled_pages(
    scope, items, total_items, total_pages, current_page
):
    with pytest.raises(ValueError, match="item count"):
        prompts_state_module.build_prompt_browse_result(
            scope,
            {
                "items": items,
                "total_items": total_items,
                "total_pages": total_pages,
                "current_page": current_page,
                "page": current_page,
                "per_page": scope.page_size,
            },
        )


def test_browse_prompt_result_deeply_freezes_detached_mapping_rows():
    source = {
        "id": "local:prompt:7",
        "local_id": 7,
        "name": "Original",
        "keywords": ["first"],
        "metadata": {"labels": ["stable"]},
    }
    result = prompts_state_module.build_prompt_browse_result(
        prompts_state_module.PromptBrowseScope(),
        {
            "items": [source],
            "total_items": 1,
            "total_pages": 1,
            "current_page": 1,
            "page": 1,
            "per_page": 20,
        },
    )

    source["name"] = "Changed"
    source["keywords"].append("late")
    source["metadata"]["labels"].append("late")

    row = result.items[0]
    assert isinstance(row, Mapping)
    assert row.get("name") == "Original"
    assert dict(row)["name"] == "Original"
    assert tuple(row) == ("id", "local_id", "name", "keywords", "metadata")
    assert row["name"] == "Original"
    assert row["keywords"] == ("first",)
    assert row["metadata"]["labels"] == ("stable",)
    with pytest.raises(TypeError):
        row["name"] = "Direct change"  # type: ignore[index]
    with pytest.raises(TypeError):
        row["metadata"]["new"] = "Direct change"  # type: ignore[index]
    with pytest.raises(TypeError):
        row["keywords"][0] = "Direct change"  # type: ignore[index]


def _direct_prompt_browse_result(items=None, **overrides):
    scope = overrides.pop("scope", prompts_state_module.PromptBrowseScope())
    values = {
        "scope": scope,
        "items": (
            [{"id": "local:prompt:8", "local_id": 8}]
            if items is None
            else items
        ),
        "total_items": 1,
        "total_pages": 1,
        "page": 1,
        "status": "ready",
        "request_fingerprint": (
            scope.fingerprint
            if isinstance(scope, prompts_state_module.PromptBrowseScope)
            else "invalid"
        ),
        "request_token": 1,
    }
    values.update(overrides)
    return prompts_state_module.PromptBrowseResult(**values)


def test_browse_prompt_result_constructor_deeply_freezes_and_detaches_items():
    source = {
        "id": "local:prompt:8",
        "local_id": 8,
        "values": [None, "text", True, 3, 4.5],
        "metadata": {
            "labels": ["stable"],
            "seen_at": datetime(2026, 8, 9, tzinfo=timezone.utc),
        },
    }
    source_items = [source]

    result = _direct_prompt_browse_result(source_items)
    source_items.append({"id": 9})
    source["values"].append("late")
    source["metadata"]["labels"].append("late")

    assert isinstance(result.items, tuple)
    assert len(result.items) == 1
    assert result.items[0]["values"] == (None, "text", True, 3, 4.5)
    assert result.items[0]["metadata"]["labels"] == ("stable",)
    assert result.items[0]["metadata"]["seen_at"] == "2026-08-09T00:00:00+00:00"
    with pytest.raises(TypeError):
        result.items[0]["id"] = 10  # type: ignore[index]


def test_browse_prompt_result_constructor_rejects_non_mapping_items():
    with pytest.raises(TypeError, match="items must be mappings"):
        _direct_prompt_browse_result(
            [{"id": "local:prompt:8", "local_id": 8}, "not a mapping"]
        )


@pytest.mark.parametrize("unsupported", [{"set value"}, object()])
def test_browse_prompt_result_rejects_unsupported_nested_leaves(unsupported):
    with pytest.raises(TypeError, match="JSON-like"):
        _direct_prompt_browse_result(
            [
                {
                    "id": "local:prompt:8",
                    "local_id": 8,
                    "unsupported": unsupported,
                }
            ]
        )


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([{"local_id": 8}], "id"),
        ([{"id": 8, "local_id": 8}], "id"),
        ([{"id": "  ", "local_id": 8}], "id"),
        ([{"id": "local:prompt:8"}], "local_id"),
        ([{"id": "local:prompt:8", "local_id": True}], "local_id"),
        ([{"id": "local:prompt:8", "local_id": 0}], "local_id"),
        (
            [
                {"id": "local:prompt:8", "local_id": 8},
                {"id": "local:prompt:8", "local_id": 9},
            ],
            "id",
        ),
        (
            [
                {"id": "local:prompt:8", "local_id": 8},
                {"id": "local:prompt:9", "local_id": 8},
            ],
            "local_id",
        ),
    ],
)
def test_browse_prompt_result_constructor_rejects_malformed_or_duplicate_identities(
    items, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        _direct_prompt_browse_result(items, total_items=len(items))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"scope": "local"}, "scope"),
        ({"total_items": True}, "total_items"),
        ({"total_items": -1}, "total_items"),
        ({"total_pages": True}, "total_pages"),
        ({"total_pages": -1}, "total_pages"),
        ({"page": True}, "page"),
        ({"page": 0}, "page"),
        ({"status": "settled"}, "status"),
        ({"request_fingerprint": None}, "request_fingerprint"),
        ({"request_fingerprint": "0" * 64}, "request_fingerprint"),
        ({"error": object()}, "error"),
    ],
)
def test_browse_prompt_result_constructor_rejects_invalid_public_fields(
    overrides, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        _direct_prompt_browse_result(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "ready", "error": "unexpected"},
        {
            "items": [],
            "total_items": 0,
            "total_pages": 0,
            "status": "error",
            "error": "",
        },
        {
            "items": [],
            "total_items": 0,
            "total_pages": 0,
            "status": "loading",
            "error": "unexpected",
        },
        {"status": "empty_library"},
        {
            "items": [],
            "total_items": 0,
            "total_pages": 0,
            "status": "ready",
        },
    ],
)
def test_browse_prompt_result_constructor_enforces_status_error_consistency(
    overrides,
):
    values = dict(overrides)
    items = values.pop("items", None)
    with pytest.raises(ValueError, match="status|error|items"):
        _direct_prompt_browse_result(items, **values)


@pytest.mark.parametrize(
    ("items", "overrides", "message"),
    [
        (
            [{"id": "local:prompt:8", "local_id": 8}],
            {"total_items": 2},
            "item count",
        ),
        (
            [{"id": "local:prompt:8", "local_id": 8}],
            {"total_pages": 2},
            "total_pages",
        ),
        (
            [{"id": "local:prompt:8", "local_id": 8}],
            {"page": 2},
            "page",
        ),
    ],
)
def test_browse_prompt_result_constructor_enforces_page_totals_and_cardinality(
    items, overrides, message
):
    with pytest.raises(ValueError, match=message):
        _direct_prompt_browse_result(items, **overrides)


def test_browse_prompt_result_constructor_accepts_valid_state_shapes():
    ready = _direct_prompt_browse_result()
    loading = _direct_prompt_browse_result(
        [], total_items=0, total_pages=0, status="loading"
    )
    error = _direct_prompt_browse_result(
        [], total_items=0, total_pages=0, status="error", error=" Retry. "
    )
    empty = _direct_prompt_browse_result(
        [], total_items=0, total_pages=0, status="empty_library"
    )

    assert ready.status == "ready"
    assert loading.status == "loading"
    assert error.error == "Retry."
    assert empty.status == "empty_library"


def test_browse_prompt_result_constructor_rejects_forged_stale_scope_guard():
    current = prompts_state_module.begin_prompt_browse(
        prompts_state_module.PromptBrowseScope(query="current"), request_token=7
    )
    stale_scope = prompts_state_module.PromptBrowseScope(query="stale")

    with pytest.raises(ValueError, match="request_fingerprint"):
        _direct_prompt_browse_result(
            [],
            scope=stale_scope,
            total_items=0,
            total_pages=0,
            status="no_matches",
            request_fingerprint=current.request_fingerprint,
            request_token=current.request_token,
        )


def test_browse_prompt_reducer_rejects_forged_stale_scope_with_copied_guards():
    current = prompts_state_module.begin_prompt_browse(
        prompts_state_module.PromptBrowseScope(query="current"), request_token=7
    )
    stale = prompts_state_module.build_prompt_browse_result(
        prompts_state_module.PromptBrowseScope(query="stale"),
        {
            "items": [],
            "total_items": 0,
            "total_pages": 0,
            "current_page": 1,
            "page": 1,
            "per_page": 20,
        },
        request_token=current.request_token,
    )
    object.__setattr__(stale, "request_fingerprint", current.request_fingerprint)

    assert prompts_state_module.apply_prompt_browse_result(current, stale) is current


@pytest.mark.parametrize(
    ("scope_kwargs", "expected_status"),
    [
        ({}, "empty_library"),
        ({"collection_id": 4}, "empty_collection"),
        ({"query": "needle"}, "no_matches"),
        ({"query": "needle", "collection_id": 4}, "no_matches"),
    ],
)
def test_browse_prompt_result_distinguishes_truthful_empty_states(
    scope_kwargs, expected_status
):
    scope = prompts_state_module.PromptBrowseScope(**scope_kwargs)
    result = prompts_state_module.build_prompt_browse_result(
        scope,
        {
            "items": [],
            "total_items": 0,
            "total_pages": 0,
            "current_page": 1,
            "page": 1,
            "per_page": scope.page_size,
        },
    )

    assert result.status == expected_status
    assert result.total_items == 0
    assert result.total_pages == 0
    assert result.page == 1


def test_browse_prompt_loading_error_and_stale_fingerprint_are_distinct():
    current_scope = prompts_state_module.PromptBrowseScope(query="current")
    loading = prompts_state_module.begin_prompt_browse(current_scope)
    error = prompts_state_module.build_prompt_browse_error(current_scope)
    stale = prompts_state_module.build_prompt_browse_result(
        prompts_state_module.PromptBrowseScope(query="stale"),
        {
            "items": [],
            "total_items": 0,
            "total_pages": 0,
            "current_page": 1,
            "page": 1,
            "per_page": 20,
        },
    )

    assert loading.status == "loading"
    assert error.status == "error"
    assert error.error == "Couldn't load prompts. Try again."
    assert prompts_state_module.apply_prompt_browse_result(loading, stale) is loading
    assert prompts_state_module.apply_prompt_browse_result(loading, error) is error


def test_browse_prompt_result_rejects_late_same_scope_request_token():
    scope = prompts_state_module.PromptBrowseScope(query="same scope")
    loading = prompts_state_module.begin_prompt_browse(scope, request_token=2)
    payload = {
        "items": [BROWSE_PROMPT_A],
        "total_items": 1,
        "total_pages": 1,
        "current_page": 1,
        "page": 1,
        "per_page": scope.page_size,
    }
    stale = prompts_state_module.build_prompt_browse_result(
        scope, payload, request_token=1
    )
    fresh = prompts_state_module.build_prompt_browse_result(
        scope, payload, request_token=2
    )

    assert prompts_state_module.apply_prompt_browse_result(loading, stale) is loading
    assert prompts_state_module.apply_prompt_browse_result(loading, fresh) is fresh
    with pytest.raises(FrozenInstanceError):
        fresh.status = "error"  # type: ignore[misc]


def test_browse_prompt_reducer_rejects_settled_state_and_loading_result():
    scope = prompts_state_module.PromptBrowseScope()
    payload = {
        "items": [BROWSE_PROMPT_A],
        "total_items": 1,
        "total_pages": 1,
        "current_page": 1,
        "page": 1,
        "per_page": scope.page_size,
    }
    settled = prompts_state_module.build_prompt_browse_result(scope, payload)
    error = prompts_state_module.build_prompt_browse_error(scope)
    loading = prompts_state_module.begin_prompt_browse(scope)
    duplicate_loading = prompts_state_module.begin_prompt_browse(scope)

    assert prompts_state_module.apply_prompt_browse_result(settled, error) is settled
    assert (
        prompts_state_module.apply_prompt_browse_result(loading, duplicate_loading)
        is loading
    )


@pytest.mark.parametrize(
    "field", ["total_items", "total_pages", "current_page", "per_page"]
)
def test_browse_prompt_result_rejects_bool_response_integers(field):
    scope = prompts_state_module.PromptBrowseScope()
    payload = {
        "items": [BROWSE_PROMPT_A],
        "total_items": 1,
        "total_pages": 1,
        "current_page": 1,
        "page": 1,
        "per_page": scope.page_size,
    }
    payload[field] = True

    with pytest.raises(ValueError, match=field):
        prompts_state_module.build_prompt_browse_result(scope, payload)


def test_browse_prompt_request_token_rejects_bool():
    scope = prompts_state_module.PromptBrowseScope()
    result = _direct_prompt_browse_result(
        [{"id": "local:prompt:8", "local_id": 8}]
    )

    with pytest.raises(ValueError, match="request_token"):
        prompts_state_module.begin_prompt_browse(scope, request_token=True)
    with pytest.raises(ValueError, match="request_token"):
        replace(result, request_token=True)


def test_browse_prompt_error_strips_text_and_rejects_whitespace_only():
    scope = prompts_state_module.PromptBrowseScope()

    result = prompts_state_module.build_prompt_browse_error(
        scope, error="  Couldn't load this page. Retry. \n"
    )

    assert result.error == "Couldn't load this page. Retry."
    with pytest.raises(ValueError, match="error"):
        prompts_state_module.build_prompt_browse_error(scope, error=" \n\t ")


def test_browse_prompt_scope_clamps_to_last_exact_page_or_first_empty_page():
    scope = prompts_state_module.PromptBrowseScope(page=9)

    assert (
        prompts_state_module.clamp_prompt_browse_scope(scope, total_pages=3).page == 3
    )
    assert (
        prompts_state_module.clamp_prompt_browse_scope(scope, total_pages=0).page == 1
    )
    assert (
        prompts_state_module.clamp_prompt_browse_scope(scope, total_pages=12) is scope
    )
    with pytest.raises(ValueError, match="total_pages"):
        prompts_state_module.clamp_prompt_browse_scope(scope, total_pages=-1)


def test_browse_prompt_list_state_preserves_service_order_and_local_identity():
    scope = prompts_state_module.PromptBrowseScope(
        sort_by="name", sort_order="desc", page_size=2
    )
    result = prompts_state_module.build_prompt_browse_result(
        scope,
        {
            "items": [
                {
                    "id": "local:prompt:uuid-z",
                    "local_id": 9,
                    "name": "Zulu",
                    "version": 5,
                    "last_modified": datetime(2020, 1, 1, tzinfo=timezone.utc),
                },
                {
                    "id": "local:prompt:uuid-a",
                    "local_id": 4,
                    "name": "Alpha",
                    "version": 2,
                    "last_modified": "2030-01-01T00:00:00+00:00",
                },
            ],
            "total_items": 2,
            "total_pages": 1,
            "current_page": 1,
            "page": 1,
            "per_page": 2,
        },
    )

    state = prompts_state_module.build_prompt_browse_list_state(result, now=NOW)

    assert [(row.prompt_id, row.name, row.version) for row in state.rows] == [
        (9, "Zulu", 5),
        (4, "Alpha", 2),
    ]
    assert state.count == 2
    assert state.sort == "name"
    assert result.items[0]["last_modified"] == "2020-01-01T00:00:00+00:00"


def test_browse_prompt_list_state_projects_separately_retained_validated_items():
    scope = prompts_state_module.PromptBrowseScope(page_size=2)
    result = prompts_state_module.build_prompt_browse_result(
        scope,
        {
            "items": [
                {
                    "id": "local:prompt:one",
                    "local_id": 1,
                    "name": "One",
                    "version": 1,
                },
                {
                    "id": "local:prompt:two",
                    "local_id": 2,
                    "name": "Two",
                    "version": 1,
                },
            ],
            "total_items": 2,
            "total_pages": 1,
            "current_page": 1,
            "page": 1,
            "per_page": 2,
        },
    )

    state = prompts_state_module.build_prompt_browse_list_state(
        result,
        now=NOW,
        retained_items=result.items[:1],
    )

    assert [row.prompt_id for row in state.rows] == [1]
    assert state.count == 1


def test_browse_prompt_list_state_rejects_unvalidated_retained_items():
    scope = prompts_state_module.PromptBrowseScope(page_size=2)
    result = prompts_state_module.build_prompt_browse_result(
        scope,
        {
            "items": [
                {
                    "id": "local:prompt:one",
                    "local_id": 1,
                    "name": "One",
                    "version": 1,
                }
            ],
            "total_items": 1,
            "total_pages": 1,
            "current_page": 1,
            "page": 1,
            "per_page": 2,
        },
    )

    with pytest.raises(ValueError, match="unique"):
        prompts_state_module.build_prompt_browse_list_state(
            result,
            now=NOW,
            retained_items=(result.items[0], result.items[0]),
        )


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        (
            {
                "local_id": True,
                "expected_version": 1,
                "title": "Title",
                "artifact_type": "prompt",
            },
            "local_id",
        ),
        (
            {
                "local_id": _PromptSelectionInt(1),
                "expected_version": 1,
                "title": "Title",
                "artifact_type": "prompt",
            },
            "local_id",
        ),
        (
            {
                "local_id": 2**63,
                "expected_version": 1,
                "title": "Title",
                "artifact_type": "prompt",
            },
            "local_id",
        ),
        (
            {
                "local_id": 1,
                "expected_version": 0,
                "title": "Title",
                "artifact_type": "prompt",
            },
            "expected_version",
        ),
        (
            {
                "local_id": 1,
                "expected_version": False,
                "title": "Title",
                "artifact_type": "prompt",
            },
            "expected_version",
        ),
        (
            {
                "local_id": 1,
                "expected_version": 1,
                "title": "",
                "artifact_type": "prompt",
            },
            "title",
        ),
        (
            {
                "local_id": 1,
                "expected_version": 1,
                "title": _PromptSelectionStr("Title"),
                "artifact_type": "prompt",
            },
            "title",
        ),
        (
            {
                "local_id": 1,
                "expected_version": 1,
                "title": "Title",
                "artifact_type": "Prompt",
            },
            "artifact_type",
        ),
    ],
)
def test_selection_entry_rejects_malformed_identity_and_display_fields(kwargs, field):
    with pytest.raises((TypeError, ValueError), match=field):
        PromptSelectionEntry(**kwargs)


@pytest.mark.parametrize("entries", [[], _PromptSelectionTuple(())])
def test_selection_basket_requires_an_exact_tuple(entries):
    entry = PromptSelectionEntry(1, 1, "One", "prompt")
    payload = entries if entries else type(entries)((entry,))

    with pytest.raises(TypeError, match="entries"):
        PromptSelectionBasket(entries=payload)


def test_selection_basket_accumulates_across_pages_and_sorts_canonical_entries():
    empty = PromptSelectionBasket()
    first = empty.select_page(
        (
            PromptSelectionEntry(9, 2, "Nine", "prompt"),
            PromptSelectionEntry(3, 4, "Three", "recipe"),
        )
    )
    second = first.select_page((PromptSelectionEntry(5, 7, "Five", "prompt"),))

    assert empty.entries == ()
    assert [entry.local_id for entry in second.entries] == [9, 3, 5]
    assert [entry.local_id for entry in second.canonical_entries] == [3, 5, 9]
    assert first.generation == 1
    assert second.generation == 2


def test_selection_select_page_preserves_existing_captured_version():
    basket = PromptSelectionBasket()
    selected = basket.toggle(PromptSelectionEntry(7, 3, "Literal [name]", "recipe"))

    same = selected.select_page((PromptSelectionEntry(7, 99, "new", "prompt"),))

    assert same is selected
    assert same.entries[0].expected_version == 3
    assert same.entries[0].title == "Literal [name]"
    assert same.entries[0].artifact_type == "recipe"
    assert same.generation == selected.generation
    assert same.canonical_entries == selected.entries


def test_selection_toggle_off_then_on_captures_the_newer_row():
    old = PromptSelectionEntry(7, 3, "Old", "prompt")
    newer = PromptSelectionEntry(7, 8, "New", "recipe")
    selected = PromptSelectionBasket().toggle(old)

    removed = selected.toggle(newer)
    reselected = removed.toggle(newer)

    assert removed.entries == ()
    assert reselected.entries == (newer,)
    assert reselected.generation == selected.generation + 2


def test_selection_select_page_suppresses_duplicates_without_generation_churn():
    entry = PromptSelectionEntry(7, 3, "Seven", "prompt")
    selected = PromptSelectionBasket().select_page((entry, entry))

    same = selected.select_page((PromptSelectionEntry(7, 9, "New", "recipe"),))

    assert selected.entries == (entry,)
    assert selected.generation == 1
    assert same is selected


def test_selection_clear_changes_generation_only_when_nonempty():
    empty = PromptSelectionBasket()
    selected = empty.toggle(PromptSelectionEntry(7, 3, "Seven", "prompt"))
    cleared = selected.clear()

    assert empty.clear() is empty
    assert cleared.entries == ()
    assert cleared.generation == selected.generation + 1
    assert cleared.clear() is cleared


@pytest.mark.parametrize(
    "page",
    [
        [PromptSelectionEntry(1, 1, "One", "prompt")],
        _PromptSelectionTuple((PromptSelectionEntry(1, 1, "One", "prompt"),)),
        (object(),),
    ],
)
def test_selection_select_page_rejects_malformed_page_shapes(page):
    basket = PromptSelectionBasket()

    with pytest.raises(TypeError, match="page"):
        basket.select_page(page)
    assert basket.entries == ()


def test_selection_browse_projection_exposes_checked_versions_and_page_counts():
    scope = prompts_state_module.PromptBrowseScope(page_size=2)
    result = prompts_state_module.build_prompt_browse_result(
        scope,
        {
            "items": [
                {
                    "id": "local:prompt:a",
                    "local_id": 7,
                    "name": "Literal [name]",
                    "version": 99,
                    "artifact_type": "recipe",
                },
                {
                    "id": "local:prompt:b",
                    "local_id": 9,
                    "name": "Nine",
                    "version": 2,
                    "artifact_type": "prompt",
                },
            ],
            "total_items": 2,
            "total_pages": 1,
            "current_page": 1,
            "page": 1,
            "per_page": 2,
        },
    )
    selection = PromptSelectionBasket(
        entries=(
            PromptSelectionEntry(7, 3, "Literal [name]", "recipe"),
            PromptSelectionEntry(11, 4, "Hidden", "prompt"),
        ),
        generation=2,
    )
    before = selection.entries

    state = prompts_state_module.build_prompt_browse_list_state(
        result, now=NOW, selection=selection, select_mode=True
    )

    assert [(row.prompt_id, row.version, row.checked) for row in state.rows] == [
        (7, 99, True),
        (9, 2, False),
    ]
    assert state.select_mode is True
    assert state.total_selected == 2
    assert state.selected_on_page == 1
    assert selection.entries is before
    assert selection.entries[0].expected_version == 3
    assert selection.generation == 2


def test_selection_browse_projection_rejects_duplicate_page_ids_before_projection():
    with pytest.raises(ValueError, match="local_id"):
        _direct_prompt_browse_result(
            [
                {
                    "id": "local:prompt:first",
                    "local_id": 7,
                    "name": "First",
                    "version": 3,
                },
                {
                    "id": "local:prompt:duplicate",
                    "local_id": 7,
                    "name": "Duplicate",
                    "version": 4,
                },
            ],
            total_items=2,
        )


@pytest.mark.parametrize(
    "malformed",
    [
        {"local_id": 2, "name": "Missing version"},
        {"local_id": 3, "name": "Bad version", "version": 0},
        {"local_id": 4, "name": "", "version": 1},
        {
            "local_id": 5,
            "name": "Bad type",
            "version": 1,
            "artifact_type": "unknown",
        },
    ],
)
def test_selection_browse_projection_rejects_malformed_page_rows(malformed):
    malformed = {
        "id": f"local:prompt:{malformed.get('local_id', 'malformed')}",
        **malformed,
    }
    result = _direct_prompt_browse_result(
        [
            {
                "id": "local:prompt:1",
                "local_id": 1,
                "name": "Valid",
                "version": 2,
            },
            malformed,
        ],
        total_items=2,
    )

    with pytest.raises(ValueError, match="project"):
        prompts_state_module.build_prompt_browse_list_state(
            result, now=NOW, selection=PromptSelectionBasket(), select_mode=True
        )


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"prompt_id": 1, "name": "One", "secondary": "", "version": 0}, "version"),
        (
            {"prompt_id": 1, "name": "One", "secondary": "", "version": True},
            "version",
        ),
        (
            {"prompt_id": 1, "name": "One", "secondary": "", "checked": 1},
            "checked",
        ),
    ],
)
def test_selection_list_row_rejects_invalid_version_or_checked(kwargs, field):
    with pytest.raises((TypeError, ValueError), match=field):
        PromptListRow(**kwargs)


def test_selection_legacy_list_keeps_convertible_ids_and_missing_versions():
    state = build_prompts_list_state(
        [
            {
                "local_id": "41",
                "id": "local:prompt:uuid",
                "name": "Legacy recipe",
                "artifact_type": "recipe",
            },
            {"id": "42", "name": None, "artifact_type": "unsupported"},
        ],
        query="",
        sort="newest",
        now=NOW,
    )

    assert [
        (row.prompt_id, row.name, row.artifact_type, row.version) for row in state.rows
    ] == [(41, "Legacy recipe", "recipe", 1), (42, "", "prompt", 1)]


def test_list_state_newest_sort_orders_by_modified_desc():
    state = build_prompts_list_state(
        [PROMPT_B, PROMPT_A], query="", sort="newest", now=NOW
    )
    assert [row.prompt_id for row in state.rows] == [1, 2]
    assert state.count == 2
    assert state.sort == "newest"


def test_list_state_name_sort_alpha_ci():
    state = build_prompts_list_state(
        [PROMPT_A, PROMPT_B], query="", sort="name", now=NOW
    )
    assert [row.name for row in state.rows] == ["brainstorm", "Summarize"]
    assert state.sort == "name"


def test_list_state_query_matches_name_case_insensitively():
    state = build_prompts_list_state(
        [PROMPT_A, PROMPT_B], query="BRAIN", sort="newest", now=NOW
    )
    assert [row.prompt_id for row in state.rows] == [2]
    assert state.count == 1


def test_list_state_query_matches_details_case_insensitively():
    """D2/U1: the filter matches ``details`` -- a field list-page records
    actually carry (unlike ``keywords``, which real list rows never do --
    see ``_prompts_page_records_or_empty``)."""
    state = build_prompts_list_state(
        [PROMPT_A, PROMPT_B], query="SUMMARIZES", sort="newest", now=NOW
    )
    assert [row.prompt_id for row in state.rows] == [1]


def test_list_state_query_does_not_silently_match_keywords_absent_from_list_rows():
    """D2/U1 regression: the old behavior matched ``keywords`` -- a field
    real list-page records never carry -- which could never actually match
    anything in production. PROMPT_A's ``keywords`` field only exists here
    because this fixture also doubles for the editor-detail-shaped tests
    below; "WRITING" (one of its keywords) is absent from every record's
    name/details, so the filter must now find nothing."""
    state = build_prompts_list_state(
        [PROMPT_A, PROMPT_B], query="WRITING", sort="newest", now=NOW
    )
    assert state.rows == ()


def test_list_state_secondary_omits_empty_details():
    state = build_prompts_list_state([PROMPT_B], query="", sort="newest", now=NOW)
    assert state.rows[0] == PromptListRow(
        prompt_id=2, name="brainstorm", secondary="1d"
    )


def test_list_state_secondary_shows_details_and_age():
    state = build_prompts_list_state([PROMPT_A], query="", sort="newest", now=NOW)
    assert state.rows[0] == PromptListRow(
        prompt_id=1,
        name="Summarize",
        secondary="Summarizes text · 3m",
        lane_summary="System + User",
        version=2,
    )


def test_list_rows_label_prompt_recipe_source_and_normalized_lane_summary():
    recipe = {
        **PROMPT_A,
        "id": 9,
        "name": "Outcome first",
        "artifact_type": "recipe",
        "backend": "server",
        "has_system_prompt": True,
        "has_user_prompt": False,
    }
    empty_prompt = {
        **PROMPT_B,
        "id": 10,
        "has_system_prompt": False,
        "has_user_prompt": False,
    }

    state = build_prompts_list_state(
        [recipe, empty_prompt], query="", sort="name", now=NOW
    )

    rows = {row.prompt_id: row for row in state.rows}
    assert rows[9].artifact_type == "recipe"
    assert rows[9].type_label == "Recipe"
    assert rows[9].source_label == "Server"
    assert rows[9].lane_summary == "System only"
    assert rows[10].type_label == "Prompt"
    assert rows[10].lane_summary == "Empty"


def test_list_state_secondary_ignores_author_and_keywords_even_when_present():
    """D2/U1: author/keywords are dropped from the secondary line entirely
    now, even when a record happens to carry them (PROMPT_C's ``author``/
    ``keywords`` here only exist because this fixture doubles for the
    editor-detail tests below) -- only details + age surface."""
    state = build_prompts_list_state([PROMPT_C], query="", sort="newest", now=NOW)
    assert state.rows[0] == PromptListRow(
        prompt_id=3, name="Zeta ideas", secondary="Ideas for the offsite · 1h"
    )


def test_editor_state_maps_fetch_prompt_details_fields():
    state = build_prompt_editor_state(PROMPT_A)
    assert (
        state.prompt_id,
        state.name,
        state.author,
        state.details,
        state.system_prompt,
        state.user_prompt,
        state.keywords_csv,
        state.version,
        state.created,
        state.modified,
    ) == (
        1,
        "Summarize",
        "Alice",
        "Summarizes text",
        "You are helpful.",
        "Summarize: {text}",
        "writing, summary",
        2,
        "",
        "2026-07-07T11:57:00+00:00",
    )
    assert state.block_editor_state is not None


def _v2_detail(*, artifact_type: str = "prompt") -> dict[str, object]:
    kind = "block_recipe" if artifact_type == "recipe" else "block_prompt"
    return {
        "id": 17,
        "name": "Structured",
        "artifact_type": artifact_type,
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": {
            "schema_version": 2,
            "kind": kind,
            "lanes": [
                {
                    "id": "system",
                    "blocks": [
                        {
                            "id": "role",
                            "title": "Role",
                            "syntax": "markdown",
                            "content": "Be precise.",
                            "mapping_hint": "Define the model's role.",
                        }
                    ],
                },
                {
                    "id": "user",
                    "blocks": [
                        {
                            "id": "goal",
                            "title": "Goal",
                            "syntax": "xml",
                            "xml_tag": "goal",
                            "content": "Ship the release.",
                        }
                    ],
                },
            ],
        },
        "system_prompt": "stale compatibility text",
        "user_prompt": "stale compatibility text",
        "version": 4,
        "backend": "local",
    }


def test_editor_state_decodes_supported_v2_into_shared_immutable_block_state():
    state = build_prompt_editor_state(_v2_detail())

    assert state.artifact_type == "prompt"
    assert state.definition_state == "supported_v2"
    assert state.block_editor_state is not None
    assert state.block_editor_state.definition.kind == "block_prompt"
    assert state.compiled_system_preview == "# Role\n\nBe precise."
    assert state.compiled_user_preview == "<goal>Ship the release.</goal>"
    assert state.compatibility_stale is True


def test_editor_state_decomposes_legacy_prompt_without_changing_lane_origins():
    detail = {
        **PROMPT_A,
        "system_prompt": "  exact system\n",
        "user_prompt": "exact user\n\n",
    }

    state = build_prompt_editor_state(detail)

    assert state.definition_state == "legacy"
    assert state.block_editor_state is not None
    assert state.block_editor_state.compiled_system == "  exact system\n"
    assert state.block_editor_state.compiled_user == "exact user\n\n"
    assert state.block_editor_state.system_origin is not None
    assert state.block_editor_state.user_origin is not None


def test_editor_state_keeps_foreign_or_malformed_artifacts_read_only_and_visible():
    detail = _v2_detail(artifact_type="recipe")
    detail["prompt_schema_version"] = 1

    state = build_prompt_editor_state(detail)

    assert state.artifact_type == "recipe"
    assert state.definition_state == "foreign_v1"
    assert state.block_editor_state is None
    assert state.compiled_system_preview == "stale compatibility text"
    assert state.can_convert_as_new is True
    assert "read-only" in state.compatibility_reason.lower()


def test_definition_state_display_label_replaces_internal_version_talk():
    """task-2859 item 2: a brand-new prompt's ``definition_state`` defaults
    to ``"legacy"`` -- the internal name for the flat-text storage format,
    not a claim the prompt is old/deprecated. The prompt editor's
    artifact-status line reads this display label, not the raw internal
    value, so a fresh prompt no longer says "legacy" verbatim."""
    assert definition_state_display_label("legacy") == "text format"
    assert definition_state_display_label("supported_v2") == "structured format"
    assert definition_state_display_label("foreign_v1") == "external format"
    # An unrecognized value still degrades gracefully (underscores spaced).
    assert definition_state_display_label("made_up_state") == "made up state"


def test_outcome_first_recipe_has_stable_blank_markdown_blocks_in_both_lanes():
    first = outcome_first_recipe()
    second = outcome_first_recipe()

    assert first == second
    assert first is not second
    assert first.kind == "block_recipe"
    assert tuple(block.id for block in first.lanes[0].blocks) == (
        "role",
        "personality",
        "collaboration-style",
    )
    assert tuple(block.id for block in first.lanes[1].blocks) == (
        "goal",
        "context-evidence",
        "constraints",
        "output",
        "success-criteria",
        "stop-rules",
    )
    assert all(
        block.syntax == "markdown"
        and block.content == ""
        and block.mapping_hint
        and block.xml_tag is None
        for lane in first.lanes
        for block in lane.blocks
    )


def test_blank_recipe_is_a_fresh_immutable_two_lane_recipe():
    first = blank_recipe()
    second = blank_recipe()

    assert first == second
    assert first is not second
    assert first.kind == "block_recipe"
    assert tuple(lane.id for lane in first.lanes) == ("system", "user")
    assert all(lane.blocks == () for lane in first.lanes)


def _draft(*, artifact_type: str = "recipe") -> PromptArtifactDraft:
    definition = outcome_first_recipe()
    if artifact_type == "prompt":
        definition = replace(definition, kind="block_prompt")
    return PromptArtifactDraft(
        artifact_type=artifact_type,  # type: ignore[arg-type]
        definition=definition,
        system_prompt="",
        user_prompt="",
        definition_bytes=b"{}",
        request_bytes=b"{}",
    )


def test_require_artifact_save_supported_accepts_exact_local_recipe_contract():
    require_artifact_save_supported(_draft(), local_prompt_capabilities())


def test_require_artifact_save_supported_rejects_type_kind_mismatch():
    draft = replace(_draft(), artifact_type="prompt")

    with pytest.raises(ValueError, match="artifact_type.*kind.*agree"):
        require_artifact_save_supported(draft, local_prompt_capabilities())


def test_require_artifact_save_supported_names_source_limit_and_recovery():
    capabilities = replace(local_prompt_capabilities(), compiled_lane_limit=3)
    draft = replace(_draft(), user_prompt="four")

    with pytest.raises(ValueError, match="user_prompt.*3 characters.*shorten"):
        require_artifact_save_supported(draft, capabilities)


def test_require_artifact_save_supported_names_definition_and_request_byte_limits():
    definition_limited = replace(local_prompt_capabilities(), definition_limit=1)
    request_limited = replace(local_prompt_capabilities(), request_limit=1)

    with pytest.raises(ValueError, match="prompt_definition.*1 UTF-8 bytes"):
        require_artifact_save_supported(_draft(), definition_limited)
    with pytest.raises(ValueError, match="request.*1 UTF-8 bytes"):
        require_artifact_save_supported(_draft(), request_limited)


def test_require_artifact_save_supported_rejects_missing_kind_capability():
    capabilities = replace(local_prompt_capabilities(), structured_kinds=frozenset())

    with pytest.raises(PromptCapabilityError, match="structured kind"):
        require_artifact_save_supported(_draft(), capabilities)


def test_require_artifact_save_supported_guards_update_version_and_capability():
    capabilities: PromptSourceCapabilities = replace(
        local_prompt_capabilities(), conditional_update=False
    )

    with pytest.raises(ValueError, match="conditional update.*save as new"):
        require_artifact_save_supported(
            _draft(), capabilities, update_original=True, expected_version=3
        )
    with pytest.raises(ValueError, match="current version.*Reload"):
        require_artifact_save_supported(
            _draft(), local_prompt_capabilities(), update_original=True
        )


HISTORY_UUID = "history-prompt-uuid"


def _history_row(
    *,
    change_id: int,
    version: int,
    prompt_uuid: str = HISTORY_UUID,
    restore_eligible: bool = True,
    compatibility_reason: str = "",
    system_preview: str = "exact system\n",
    user_preview: str = "exact user\n",
    keywords_captured: bool = True,
) -> dict[str, object]:
    return {
        "prompt_uuid": prompt_uuid,
        "change_id": change_id,
        "version": version,
        "timestamp": f"2026-08-08T12:00:0{version}+00:00",
        "artifact_type": "prompt",
        "name": f"Version {version}",
        "author": "Author",
        "details": "Literal metadata",
        "compiled_system_prompt": system_preview,
        "compiled_user_prompt": user_preview,
        "keywords": ["alpha", "beta"],
        "keywords_captured": keywords_captured,
        "compatibility_state": "compatible" if restore_eligible else "foreign_v1",
        "compatibility_reason": compatibility_reason,
        "restore_eligible": restore_eligible,
        "changed_fields": ["system_prompt"],
        "change_summary": "System prompt",
    }


def _history_page(
    *items: dict[str, object],
    total_count: int,
    has_more: bool,
    next_before_change_id: int | None,
) -> dict[str, object]:
    return {
        "items": list(items),
        "total_count": total_count,
        "has_more": has_more,
        "next_before_change_id": next_before_change_id,
    }


def _normalized_history_row(payload: dict[str, object]) -> dict[str, object]:
    """Produce the real normalized retained-history row shape consumed by state."""
    return normalize_prompt_history_page(
        {
            "items": [
                {
                    "change_id": 30,
                    "entity": "Prompts",
                    "entity_uuid": HISTORY_UUID,
                    "operation": "update",
                    "timestamp": "2026-08-08T12:00:03+00:00",
                    "version": 3,
                    "payload": payload,
                }
            ],
            "predecessor": None,
            "total_count": 1,
            "has_more": False,
            "next_before_change_id": None,
        },
        backend="local",
    )["items"][0]


def test_history_row_uses_literal_stored_lanes_when_normalized_preview_mismatches():
    normalized = _normalized_history_row(
        {
            "name": "Mismatch",
            "author": None,
            "details": "stored metadata",
            "system_prompt": "  stored system [literal]\n",
            "user_prompt": "stored user\n\n",
            "prompt_format": "structured",
            "prompt_schema_version": 2,
            "prompt_definition": {
                "schema_version": 2,
                "kind": "block_prompt",
                "lanes": [
                    {
                        "id": "system",
                        "blocks": [
                            {
                                "id": "role",
                                "title": "Role",
                                "syntax": "freeform",
                                "content": "definition-derived system",
                            }
                        ],
                    },
                    {
                        "id": "user",
                        "blocks": [
                            {
                                "id": "request",
                                "title": "Request",
                                "syntax": "freeform",
                                "content": "definition-derived user",
                            }
                        ],
                    },
                ],
            },
            "artifact_type": "prompt",
            "keywords": [],
        }
    )
    assert normalized["compatibility_state"] == "compiled_text_mismatch"
    assert normalized["compiled_system_prompt"] != normalized["system_prompt"]

    row = build_prompt_history_page(
        _history_page(
            normalized,
            total_count=1,
            has_more=False,
            next_before_change_id=None,
        )
    ).items[0]

    assert row.system_preview == "  stored system [literal]\n"
    assert row.user_preview == "stored user\n\n"

    fallback = _history_row(change_id=29, version=2)
    fallback["compiled_system_prompt"] = "compiled fallback"
    fallback["compiled_user_prompt"] = "compiled fallback user"
    fallback_row = build_prompt_history_page(
        _history_page(
            fallback,
            total_count=1,
            has_more=False,
            next_before_change_id=None,
        )
    ).items[0]
    assert fallback_row.system_preview == "compiled fallback"
    assert fallback_row.user_preview == "compiled fallback user"


def test_history_row_preserves_normalized_unsupported_artifact_raw_identity():
    normalized = _normalized_history_row(
        {
            "name": "Future",
            "author": None,
            "details": "",
            "system_prompt": "stored system",
            "user_prompt": "stored user",
            "prompt_format": "structured",
            "prompt_schema_version": 2,
            "prompt_definition": None,
            "artifact_type": "future-artifact",
            "keywords": [],
        }
    )
    assert normalized["artifact_type"] == "unsupported"
    assert normalized["artifact_type_raw"] == "future-artifact"

    row = build_prompt_history_page(
        _history_page(
            normalized,
            total_count=1,
            has_more=False,
            next_before_change_id=None,
        )
    ).items[0]

    assert row.artifact_type == "unsupported"
    assert row.artifact_type_raw == "future-artifact"
    assert row.restore_eligible is False


def test_history_state_starts_closed_then_counts_and_loads_first_page():
    state = build_prompt_history_state(
        prompt_uuid=HISTORY_UUID, current_version=4, scope_token=10
    )

    assert state.page_status == "closed"
    assert state.count_status == "idle"
    assert prompt_history_count_label(state) == "Retained history (…)"

    state, count_request = begin_prompt_history_count(state, request_token=11)
    assert state.count_status == "loading"
    assert count_request.prompt_uuid == HISTORY_UUID
    state = apply_prompt_history_count(state, count_request, total_count=7)

    assert state.count_status == "loaded"
    assert state.retained_count == 7
    assert prompt_history_count_label(state) == "Retained history (7)"

    state, page_request = begin_prompt_history_page(state, request_token=12)
    assert state.is_open is True
    assert state.page_status == "loading"
    state = apply_prompt_history_page(
        state,
        page_request,
        build_prompt_history_page(
            _history_page(
                _history_row(change_id=40, version=4),
                _history_row(change_id=30, version=3),
                total_count=7,
                has_more=True,
                next_before_change_id=30,
            )
        ),
    )

    assert state.page_status == "loaded"
    assert [row.version for row in state.rows] == [4, 3]
    assert state.retained_count == 7


def test_history_page_error_can_retry_without_discarding_identity_or_count():
    state = build_prompt_history_state(
        prompt_uuid=HISTORY_UUID, current_version=4, scope_token=10
    )
    state, request = begin_prompt_history_page(state, request_token=11)
    state = apply_prompt_history_page(state, request, None, error="Network unavailable")

    assert state.page_status == "error"
    assert state.error == "Network unavailable"
    assert state.prompt_uuid == HISTORY_UUID
    state, retry = begin_prompt_history_page(state, request_token=12)

    assert state.page_status == "loading"
    assert retry.before_change_id is None


def test_closing_history_returns_to_closed_state_and_rejects_its_late_page():
    state = build_prompt_history_state(
        prompt_uuid=HISTORY_UUID, current_version=4, scope_token=10
    )
    state, request = begin_prompt_history_page(state, request_token=11)
    closed = close_prompt_history(state)

    assert closed.is_open is False
    assert closed.page_status == "closed"
    assert (
        apply_prompt_history_page(
            closed,
            request,
            build_prompt_history_page(
                _history_page(
                    _history_row(change_id=40, version=4),
                    total_count=1,
                    has_more=False,
                    next_before_change_id=None,
                )
            ),
        )
        == closed
    )


def test_newer_count_request_is_not_replaced_by_an_older_page_result():
    state = build_prompt_history_state(
        prompt_uuid=HISTORY_UUID, current_version=4, scope_token=10
    )
    state, page_request = begin_prompt_history_page(state, request_token=11)
    state, count_request = begin_prompt_history_count(state, request_token=12)

    state = apply_prompt_history_page(
        state,
        page_request,
        build_prompt_history_page(
            _history_page(
                _history_row(change_id=40, version=4),
                total_count=4,
                has_more=False,
                next_before_change_id=None,
            )
        ),
    )

    assert state.count_status == "loading"
    assert state.count_request == count_request
    state = apply_prompt_history_count(state, count_request, total_count=7)
    assert state.retained_count == 7


def test_history_older_pages_append_in_order_and_reject_duplicate_rows():
    state = build_prompt_history_state(
        prompt_uuid=HISTORY_UUID, current_version=4, scope_token=10
    )
    state, newest_request = begin_prompt_history_page(state, request_token=11)
    state = apply_prompt_history_page(
        state,
        newest_request,
        build_prompt_history_page(
            _history_page(
                _history_row(change_id=40, version=4),
                _history_row(change_id=30, version=3),
                total_count=4,
                has_more=True,
                next_before_change_id=30,
            )
        ),
    )
    state, older_request = begin_prompt_history_page(state, request_token=12)
    assert older_request.before_change_id == 30
    state = apply_prompt_history_page(
        state,
        older_request,
        build_prompt_history_page(
            _history_page(
                _history_row(change_id=20, version=2),
                _history_row(change_id=10, version=1),
                total_count=4,
                has_more=False,
                next_before_change_id=None,
            )
        ),
    )

    assert [row.change_id for row in state.rows] == [40, 30, 20, 10]
    assert state.next_before_change_id is None

    with pytest.raises(ValueError, match="No older retained history pages"):
        begin_prompt_history_page(state, request_token=13)
    with pytest.raises(ValueError, match="duplicate change IDs"):
        build_prompt_history_page(
            _history_page(
                _history_row(change_id=10, version=1),
                _history_row(change_id=10, version=1),
                total_count=4,
                has_more=False,
                next_before_change_id=None,
            )
        )


def _loaded_history_state(*, has_more: bool = False) -> object:
    state = build_prompt_history_state(
        prompt_uuid=HISTORY_UUID, current_version=4, scope_token=10
    )
    state, page_request = begin_prompt_history_page(state, request_token=11)
    return apply_prompt_history_page(
        state,
        page_request,
        build_prompt_history_page(
            _history_page(
                _history_row(change_id=40, version=4),
                _history_row(change_id=30, version=3),
                total_count=4,
                has_more=has_more,
                next_before_change_id=30 if has_more else None,
            )
        ),
    )


def test_matching_overlapping_older_page_settles_error_and_preserves_loaded_rows():
    state = _loaded_history_state(has_more=True)
    state, preview_request = begin_prompt_history_preview(
        state, change_id=30, source_version=3, request_token=12
    )
    state = apply_prompt_history_preview(state, preview_request)
    selected = state.selected
    state, request = begin_prompt_history_page(state, request_token=13)
    before_rows = state.rows

    settled = apply_prompt_history_page(
        state,
        request,
        build_prompt_history_page(
            _history_page(
                _history_row(change_id=30, version=3),
                total_count=4,
                has_more=False,
                next_before_change_id=None,
            )
        ),
    )

    assert settled.page_status == "error"
    assert settled.page_request is None
    assert settled.rows == before_rows
    assert settled.selected == selected
    assert settled.error == "Retained history page overlaps an already loaded row."


def test_matching_cursor_mismatch_settles_error_and_preserves_loaded_rows():
    state = _loaded_history_state(has_more=True)
    state, request = begin_prompt_history_page(state, request_token=12)
    invalidated = replace(state, next_before_change_id=29)

    settled = apply_prompt_history_page(
        invalidated,
        request,
        build_prompt_history_page(
            _history_page(
                _history_row(change_id=20, version=2),
                total_count=4,
                has_more=False,
                next_before_change_id=None,
            )
        ),
    )

    assert settled.page_status == "error"
    assert settled.page_request is None
    assert settled.rows == state.rows
    assert settled.error == "Retained history page cursor no longer matches."


def test_stale_invalid_page_result_does_nothing_while_current_request_stays_loading():
    state = _loaded_history_state(has_more=True)
    state, stale_request = begin_prompt_history_page(state, request_token=12)
    state, current_request = begin_prompt_history_page(state, request_token=13)
    invalid_page = build_prompt_history_page(
        _history_page(
            _history_row(change_id=30, version=3),
            total_count=4,
            has_more=False,
            next_before_change_id=None,
        )
    )

    assert apply_prompt_history_page(state, stale_request, invalid_page) == state
    assert state.page_request == current_request
    assert state.page_status == "loading"


def test_matching_missing_preview_row_clears_request_and_keeps_prior_selection():
    state = _loaded_history_state()
    state, first_request = begin_prompt_history_preview(
        state, change_id=30, source_version=3, request_token=12
    )
    state = apply_prompt_history_preview(state, first_request)
    selected = state.selected
    state, request = begin_prompt_history_preview(
        state, change_id=30, source_version=3, request_token=13
    )
    invalidated = replace(state, rows=())

    settled = apply_prompt_history_preview(invalidated, request)

    assert settled.preview_request is None
    assert settled.selected == selected
    assert settled.error == "Selected retained version is no longer loaded."


def test_matching_restore_request_uses_service_outcome_after_preview_ui_changes():
    """The accepted write request, not transient preview UI, owns settlement."""
    state = _loaded_history_state()
    state, preview_request = begin_prompt_history_preview(
        state, change_id=30, source_version=3, request_token=12
    )
    state = apply_prompt_history_preview(state, preview_request)
    state, request, _ = begin_prompt_history_restore(
        state, request_token=13, dirty=False
    )
    assert request is not None
    selected = state.selected
    current_changed = replace(state, current_version=5)
    outcome = format_prompt_history_restore_outcome(error=RuntimeError("ignored"))

    settled_current = apply_prompt_history_restore(current_changed, request, outcome)

    assert settled_current.restore_request is None
    assert settled_current.selected == selected
    assert settled_current.restore_outcome == outcome

    state = _loaded_history_state()
    state, preview_request = begin_prompt_history_preview(
        state, change_id=30, source_version=3, request_token=12
    )
    state = apply_prompt_history_preview(state, preview_request)
    state, request, _ = begin_prompt_history_restore(
        state, request_token=13, dirty=False
    )
    assert request is not None and state.selected is not None
    changed_selection = replace(
        state, selected=replace(state.selected, source_version=2)
    )

    settled_selection = apply_prompt_history_restore(
        changed_selection, request, outcome
    )

    assert settled_selection.restore_request is None
    assert settled_selection.selected == changed_selection.selected
    assert settled_selection.restore_outcome == outcome


def test_closing_history_clears_active_preview_request():
    state = _loaded_history_state()
    state, preview_request = begin_prompt_history_preview(
        state, change_id=30, source_version=3, request_token=12
    )
    closed_preview = close_prompt_history(state)

    assert closed_preview.preview_request is None
    assert (
        apply_prompt_history_preview(closed_preview, preview_request) == closed_preview
    )


def test_closing_history_preserves_and_applies_an_active_conditional_restore():
    """Collapse clears preview UI without cancelling an accepted DB write."""
    state = _loaded_history_state()
    state, preview_request = begin_prompt_history_preview(
        state, change_id=30, source_version=3, request_token=12
    )
    state = apply_prompt_history_preview(state, preview_request)
    state, restore_request, _ = begin_prompt_history_restore(
        state, request_token=13, dirty=False
    )
    assert restore_request is not None

    closed = close_prompt_history(state)
    outcome = format_prompt_history_restore_outcome(
        {
            "outcome": "restored",
            "source_version": 3,
            "current_version": 4,
            "new_version": 5,
            "retained_current_keywords": False,
        }
    )
    settled = apply_prompt_history_restore(closed, restore_request, outcome)

    assert closed.rows == ()
    assert closed.selected is None
    assert closed.restore_request == restore_request
    assert settled.restore_request is None
    assert settled.restore_outcome == outcome
    assert settled.restore_refresh_pending is True


def test_reload_history_page_clears_page_scope_but_preserves_settled_count():
    state = replace(
        _loaded_history_state(),
        restore_outcome=PromptHistoryRestoreOutcome(
            kind="snapshot_unavailable",
            message="Reload retained history.",
            reload_required=True,
        ),
    )

    reset = reset_prompt_history_page(state)

    assert reset.is_open is True
    assert reset.page_status == "closed"
    assert reset.rows == ()
    assert reset.selected is None
    assert reset.restore_outcome is None
    assert reset.retained_count == state.retained_count
    assert reset.count_status == state.count_status
    assert reset.current_version == state.current_version


@pytest.mark.parametrize(
    ("result", "error", "kind"),
    [
        (
            {
                "outcome": "no_change",
                "source_version": 3,
                "current_version": 4,
                "new_version": 4,
            },
            None,
            "no_change",
        ),
        (
            None,
            PromptRestoreError(PromptRestoreErrorCode.EXPECTED_VERSION),
            "conflict",
        ),
        (
            {"outcome": "snapshot_unavailable", "source_version": 3},
            None,
            "snapshot_unavailable",
        ),
        (
            {"outcome": "current_unavailable", "source_version": 3},
            None,
            "current_unavailable",
        ),
        (
            None,
            PromptRestoreError(PromptRestoreErrorCode.VALIDATION),
            "validation_error",
        ),
        (
            None,
            PromptRestoreError(PromptRestoreErrorCode.NAME_CONFLICT),
            "name_conflict",
        ),
        (None, ValueError("SECRET adapter value"), "error"),
        (None, ConflictError("SECRET unclassified conflict"), "error"),
        (None, RuntimeError("network"), "error"),
    ],
)
def test_restore_non_success_outcomes_keep_selected_row_retryable(
    result: dict[str, object] | None, error: Exception | None, kind: str
):
    state = _loaded_history_state()
    state, preview_request = begin_prompt_history_preview(
        state, change_id=30, source_version=3, request_token=12
    )
    state = apply_prompt_history_preview(state, preview_request)
    selected = state.selected
    state, request, _ = begin_prompt_history_restore(
        state, request_token=13, dirty=False
    )
    assert request is not None

    settled = apply_prompt_history_restore(
        state, request, format_prompt_history_restore_outcome(result, error=error)
    )

    assert settled.restore_request is None
    assert settled.restore_outcome is not None
    assert settled.restore_outcome.kind == kind
    if kind == "error":
        assert settled.restore_outcome.message == "Couldn't restore retained history."
        assert "SECRET" not in settled.restore_outcome.message
    assert settled.selected == selected
    assert history_restore_gate(settled, dirty=False).enabled is True


def test_history_preview_selection_preserves_literal_preview_and_explicit_source_version():
    state = build_prompt_history_state(
        prompt_uuid=HISTORY_UUID, current_version=4, scope_token=10
    )
    state, page_request = begin_prompt_history_page(state, request_token=11)
    state = apply_prompt_history_page(
        state,
        page_request,
        build_prompt_history_page(
            _history_page(
                _history_row(
                    change_id=30,
                    version=3,
                    system_preview="  literal system\n",
                    user_preview="literal user\n\n",
                ),
                total_count=1,
                has_more=False,
                next_before_change_id=None,
            )
        ),
    )
    state, preview_request = begin_prompt_history_preview(
        state, change_id=30, source_version=3, request_token=12
    )
    state = apply_prompt_history_preview(state, preview_request)

    assert state.selected is not None
    assert state.selected.prompt_uuid == HISTORY_UUID
    assert state.selected.change_id == 30
    assert state.selected.source_version == 3
    assert state.selected.row.system_preview == "  literal system\n"
    assert state.selected.row.user_preview == "literal user\n\n"


def test_history_restore_gate_captures_identity_versions_and_dirty_compatibility_reasons():
    state = build_prompt_history_state(
        prompt_uuid=HISTORY_UUID, current_version=4, scope_token=10
    )
    state, page_request = begin_prompt_history_page(state, request_token=11)
    state = apply_prompt_history_page(
        state,
        page_request,
        build_prompt_history_page(
            _history_page(
                _history_row(change_id=30, version=3),
                total_count=1,
                has_more=False,
                next_before_change_id=None,
            )
        ),
    )
    state, preview_request = begin_prompt_history_preview(
        state, change_id=30, source_version=3, request_token=12
    )
    state = apply_prompt_history_preview(state, preview_request)

    dirty_gate = history_restore_gate(state, dirty=True)
    assert dirty_gate.enabled is False
    assert (
        dirty_gate.reason
        == "Save or discard unsaved changes before restoring retained history."
    )

    gate = history_restore_gate(state, dirty=False)
    assert gate.enabled is True
    assert gate.target is not None
    assert (
        gate.target.prompt_uuid,
        gate.target.change_id,
        gate.target.source_version,
        gate.target.expected_current_version,
    ) == (HISTORY_UUID, 30, 3, 4)

    incompatible = build_prompt_history_state(
        prompt_uuid=HISTORY_UUID, current_version=4, scope_token=20
    )
    incompatible, request = begin_prompt_history_page(incompatible, request_token=21)
    incompatible = apply_prompt_history_page(
        incompatible,
        request,
        build_prompt_history_page(
            _history_page(
                _history_row(
                    change_id=20,
                    version=2,
                    restore_eligible=False,
                    compatibility_reason="Foreign v1 retained artifacts are preview-only.",
                ),
                total_count=1,
                has_more=False,
                next_before_change_id=None,
            )
        ),
    )
    incompatible, request = begin_prompt_history_preview(
        incompatible, change_id=20, source_version=2, request_token=22
    )
    incompatible = apply_prompt_history_preview(incompatible, request)

    compatibility_gate = history_restore_gate(incompatible, dirty=False)
    assert compatibility_gate.enabled is False
    assert (
        compatibility_gate.reason == "Foreign v1 retained artifacts are preview-only."
    )


def test_history_restore_outcomes_have_stable_copy_and_keyword_disclosure():
    restored = format_prompt_history_restore_outcome(
        {
            "outcome": "restored",
            "source_version": 2,
            "current_version": 4,
            "new_version": 5,
            "retained_current_keywords": True,
        }
    )
    assert restored == PromptHistoryRestoreOutcome(
        kind="restored",
        message="Restored v2 as current v5.",
        reload_required=False,
        keyword_disclosure=(
            "Current keywords were retained because this older retained version "
            "did not capture keywords."
        ),
    )
    assert (
        format_prompt_history_restore_outcome(
            {
                "outcome": "no_change",
                "source_version": 2,
                "current_version": 4,
                "new_version": 4,
                "retained_current_keywords": False,
            }
        ).message
        == "Retained v2 already matches current v4; no new version was created."
    )
    assert (
        format_prompt_history_restore_outcome(
            {"outcome": "snapshot_unavailable", "source_version": 2}
        ).reload_required
        is True
    )
    assert (
        format_prompt_history_restore_outcome(
            {"outcome": "current_unavailable", "source_version": 2}
        ).kind
        == "current_unavailable"
    )
    assert format_prompt_history_restore_outcome(
        error=PromptRestoreError(PromptRestoreErrorCode.EXPECTED_VERSION)
    ).message == ("This Prompt changed elsewhere. Reload before restoring.")
    assert (
        format_prompt_history_restore_outcome(
            error=PromptRestoreError(PromptRestoreErrorCode.VALIDATION)
        ).kind
        == "validation_error"
    )
    generic = format_prompt_history_restore_outcome(
        error=ValueError("SECRET validation payload")
    )
    assert generic.kind == "error"
    assert generic.message == "Couldn't restore retained history."
    assert "SECRET" not in generic.message


def test_stale_history_count_page_selection_and_restore_outcomes_are_ignored():
    state = build_prompt_history_state(
        prompt_uuid=HISTORY_UUID, current_version=4, scope_token=10
    )
    state, old_count = begin_prompt_history_count(state, request_token=11)
    state, new_count = begin_prompt_history_count(state, request_token=12)
    assert apply_prompt_history_count(state, old_count, total_count=99) == state
    state = apply_prompt_history_count(state, new_count, total_count=4)

    state, page_request = begin_prompt_history_page(state, request_token=13)
    page = build_prompt_history_page(
        _history_page(
            _history_row(change_id=30, version=3),
            total_count=4,
            has_more=False,
            next_before_change_id=None,
        )
    )
    stale_scope = build_prompt_history_state(
        prompt_uuid=HISTORY_UUID, current_version=4, scope_token=20
    )
    assert apply_prompt_history_page(stale_scope, page_request, page) == stale_scope
    state = apply_prompt_history_page(state, page_request, page)
    state, old_preview = begin_prompt_history_preview(
        state, change_id=30, source_version=3, request_token=14
    )
    state, current_preview = begin_prompt_history_preview(
        state, change_id=30, source_version=3, request_token=15
    )
    assert apply_prompt_history_preview(state, old_preview) == state
    state = apply_prompt_history_preview(state, current_preview)
    state, restore_request, gate = begin_prompt_history_restore(
        state, request_token=16, dirty=False
    )
    assert gate.enabled is True
    assert restore_request is not None
    state, newer_restore_request, _ = begin_prompt_history_restore(
        state, request_token=17, dirty=False
    )
    assert newer_restore_request is not None
    stale_outcome = format_prompt_history_restore_outcome(
        {
            "outcome": "restored",
            "source_version": 3,
            "current_version": 4,
            "new_version": 5,
            "retained_current_keywords": False,
        }
    )
    assert apply_prompt_history_restore(state, restore_request, stale_outcome) == state
    settled = apply_prompt_history_restore(state, newer_restore_request, stale_outcome)
    assert settled.restore_outcome == stale_outcome


def test_prepare_recipe_save_defaults_to_empty_content_and_preserves_structure():
    definition = outcome_first_recipe()
    populated = replace(
        definition,
        lanes=(
            replace(
                definition.lanes[0],
                blocks=(replace(definition.lanes[0].blocks[0], content="Architect"),),
            ),
            definition.lanes[1],
        ),
    )
    state = build_prompt_editor_state(
        {
            "artifact_type": "recipe",
            "prompt_format": "structured",
            "prompt_schema_version": 2,
            "prompt_definition": {
                "kind": populated.kind,
                "schema_version": populated.schema_version,
                "lanes": [
                    {
                        "id": lane.id,
                        "blocks": [
                            {
                                "id": block.id,
                                "title": block.title,
                                "syntax": block.syntax,
                                "content": block.content,
                                "mapping_hint": block.mapping_hint,
                            }
                            for block in lane.blocks
                        ],
                    }
                    for lane in populated.lanes
                ],
            },
        }
    ).block_editor_state
    assert state is not None

    draft, payload, saved_state = prepare_prompt_artifact_save(
        state,
        artifact_type="recipe",
        include_recipe_starter_content=False,
        request_fields={"name": "Outcome first", "keywords": None},
    )

    assert draft.artifact_type == "recipe"
    assert draft.system_prompt == ""
    assert all(
        block.content == ""
        for lane in saved_state.definition.lanes
        for block in lane.blocks
    )
    assert saved_state.definition.lanes[0].blocks[0].title == "Role"
    assert (
        saved_state.definition.lanes[0].blocks[0].mapping_hint
        == "Define the model's function and job."
    )
    assert payload["artifact_type"] == "recipe"
    assert "keywords" not in payload
    assert payload["prompt_definition"]["kind"] == "block_recipe"
    assert draft.definition_bytes
    assert draft.request_bytes


def test_prepare_recipe_save_preserves_content_only_when_explicitly_selected():
    state = build_prompt_editor_state(
        {"system_prompt": "Stay direct.", "user_prompt": "Draft the plan."}
    ).block_editor_state
    assert state is not None

    draft, payload, saved_state = prepare_prompt_artifact_save(
        state,
        artifact_type="recipe",
        include_recipe_starter_content=True,
        request_fields={"name": "Planning recipe"},
    )

    assert draft.system_prompt == "Stay direct."
    assert draft.user_prompt == "Draft the plan."
    assert saved_state.artifact_type == "recipe"
    assert payload["prompt_definition"]["kind"] == "block_recipe"


def test_editor_state_resolves_prompt_id_from_local_id_when_id_is_composite_string():
    """Critical regression: the REAL production seam
    (``PromptScopeService.get_prompt`` -> ``normalize_prompt_record``, see
    ``tldw_chatbook/Prompt_Management/prompt_normalizers.py``) returns
    ``detail["id"]`` as the COMPOSITE STRING ``"<backend>:prompt:<uuid>"``
    -- the raw local numeric id lives under ``detail["local_id"]`` instead.
    ``_to_int`` silently swallows the ``ValueError`` on the composite
    string, so ``build_prompt_editor_state`` used to return
    ``prompt_id=None`` for every EXISTING saved prompt loaded this way,
    which made ``prompt_editor_meta_line`` render "New prompt" instead of
    "Modified ... · vN". ``build_prompt_editor_state`` must prefer
    ``local_id`` when present."""
    detail = {
        "id": "local:prompt:9f4e2f0a-1111-2222-3333-444455556666",
        "backend": "local",
        "source_id": "9f4e2f0a-1111-2222-3333-444455556666",
        "local_id": 7,
        "server_id": None,
        "uuid": "9f4e2f0a-1111-2222-3333-444455556666",
        "name": "Summarize",
        "author": "Alice",
        "details": "Summarizes text",
        "system_prompt": "You are helpful.",
        "user_prompt": "Summarize: {text}",
        "keywords": ["writing", "summary"],
        "version": 2,
        "last_modified": "2026-07-07T11:57:00+00:00",
    }
    state = build_prompt_editor_state(detail)
    assert state.prompt_id == 7
    assert prompt_editor_meta_line(state, now=NOW) == "Modified 3m · v2"


def test_editor_state_prompt_id_none_when_local_id_absent_and_id_is_composite_string():
    """The server-backend shape (``local_id`` present but ``None``, ``id``
    a composite string) must still resolve to ``prompt_id=None`` rather
    than raising -- unchanged from before this fix (server prompts were
    never resolvable via the plain ``id`` field either)."""
    detail = {
        "id": "server:prompt:9f4e2f0a-1111-2222-3333-444455556666",
        "backend": "server",
        "local_id": None,
        "server_id": 7,
        "name": "Summarize",
    }
    state = build_prompt_editor_state(detail)
    assert state.prompt_id is None


def test_editor_state_prompt_id_none_for_blank_create_flow_detail():
    """The D1 blank-create / Duplicate-action detail shapes
    (``_enter_library_prompt_create_editor``,
    ``handle_library_prompt_duplicate``) never carry an ``id`` or
    ``local_id`` key at all -- ``prompt_id`` must stay ``None`` so the
    editor still renders "New prompt", not a false "Modified ... · vN"."""
    detail = {
        "name": "Brand New (copy)",
        "author": "Alice",
        "details": "d",
        "system_prompt": "s",
        "user_prompt": "u",
        "keywords": "kw1, kw2",
    }
    state = build_prompt_editor_state(detail)
    assert state.prompt_id is None
    assert prompt_editor_meta_line(state) == "New prompt"


def test_editor_state_tolerates_empty_mapping():
    state = build_prompt_editor_state({})
    assert (
        state.prompt_id,
        state.name,
        state.author,
        state.details,
        state.system_prompt,
        state.user_prompt,
        state.keywords_csv,
        state.version,
        state.created,
        state.modified,
    ) == (None, "", "", "", "", "", "", None, "", "")
    assert state.block_editor_state is not None


def test_classify_soft_deleted_name():
    message = (
        "Prompt 'Foo' exists but is soft-deleted. Use overwrite to restore/update."
    )
    assert classify_prompt_save_error(None, message, None) == "soft-deleted-name"


def test_classify_conflict_error():
    assert classify_prompt_save_error(None, "", ConflictError("x")) == "conflict"


def test_classify_name_in_use_from_integrity_error():
    exc = sqlite3.IntegrityError("UNIQUE constraint failed: Prompts.name")
    assert classify_prompt_save_error(None, "", exc) == "name-in-use"


def test_classify_ok():
    assert classify_prompt_save_error(5, "", None) == "ok"


def test_classify_error_fallback():
    assert classify_prompt_save_error(None, "boom", RuntimeError("boom")) == "error"


def test_meta_line_new_prompt_sentinel_overrides_modified_and_version():
    """Task 8b D1: a blank, not-yet-saved editor state (``prompt_id=None``)
    renders "New prompt", never "Modified … · vN" -- even when the caller
    (a malformed record) happens to also carry ``modified``/``version``."""
    state = build_prompt_editor_state(
        {"last_modified": "2026-07-07T11:00:00+00:00", "version": 3}
    )
    assert state.prompt_id is None
    assert prompt_editor_meta_line(state) == "New prompt"


def test_meta_line_existing_prompt_unaffected_by_new_prompt_sentinel():
    state = build_prompt_editor_state(PROMPT_A)
    assert prompt_editor_meta_line(state, now=NOW) == "Modified 3m · v2"


def test_meta_line_appends_unsaved_marker_when_dirty():
    """U6 (Task 8c): a dirty editor's meta line gets a trailing unsaved
    marker -- ``dirty`` is a plain pure-function input, not derived from
    ``PromptEditorState`` itself."""
    state = build_prompt_editor_state(PROMPT_A)
    assert prompt_editor_meta_line(state, now=NOW, dirty=True) == (
        "Modified 3m · v2 · • Unsaved changes"
    )


def test_meta_line_omits_unsaved_marker_when_not_dirty():
    """``dirty`` defaults to ``False`` -- existing callers that never pass
    it keep the exact same rendering as before this change."""
    state = build_prompt_editor_state(PROMPT_A)
    assert prompt_editor_meta_line(state, now=NOW, dirty=False) == "Modified 3m · v2"
    assert prompt_editor_meta_line(state, now=NOW) == "Modified 3m · v2"


def test_meta_line_new_prompt_sentinel_appends_unsaved_marker_when_dirty():
    """The "New prompt" sentinel also gets the unsaved marker once the user
    starts typing into a blank create-flow record (dirty becomes True)."""
    state = build_prompt_editor_state({})
    assert (
        prompt_editor_meta_line(state, dirty=True) == "New prompt · • Unsaved changes"
    )
    assert prompt_editor_meta_line(state) == "New prompt"


def _basic_structured_prompt(*, extra_user_block: bool = False):
    user_blocks = [
        {
            "id": "message",
            "title": "Message",
            "syntax": "freeform",
            "content": "Summarize this.",
        }
    ]
    if extra_user_block:
        user_blocks.append(
            {
                "id": "audience",
                "title": "Audience",
                "syntax": "xml",
                "xml_tag": "audience",
                "content": "Executives",
            }
        )
    return build_prompt_editor_state(
        {
            "id": 44,
            "artifact_type": "prompt",
            "prompt_format": "structured",
            "prompt_schema_version": 2,
            "prompt_definition": {
                "schema_version": 2,
                "kind": "block_prompt",
                "lanes": [
                    {
                        "id": "system",
                        "blocks": [
                            {
                                "id": "instructions",
                                "title": "Instructions",
                                "syntax": "markdown",
                                "content": "Be concise.",
                            }
                        ],
                    },
                    {"id": "user", "blocks": user_blocks},
                ],
            },
            "system_prompt": "# Instructions\n\nBe concise.",
            "user_prompt": (
                "Summarize this.\n\n<audience>Executives</audience>"
                if extra_user_block
                else "Summarize this."
            ),
            "version": 2,
        }
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "basic"),
        ("", "basic"),
        ("BASIC", "basic"),
        ("advanced", "advanced"),
        ("future-mode", "basic"),
        (True, "basic"),
    ],
)
def test_prompt_editor_mode_coercion_fails_to_basic(raw, expected):
    assert coerce_prompt_editor_mode(raw) == expected


def test_prompt_basic_accepts_legacy_and_one_block_per_lane_without_mutation():
    legacy = build_prompt_editor_state(PROMPT_A)
    structured = _basic_structured_prompt()
    legacy_block_state = legacy.block_editor_state
    structured_block_state = structured.block_editor_state

    assert prompt_basic_unavailable_reason(legacy) == ""
    assert prompt_basic_unavailable_reason(structured) == ""
    assert legacy.block_editor_state is legacy_block_state
    assert structured.block_editor_state is structured_block_state


@pytest.mark.parametrize(
    ("state", "kwargs", "expected"),
    [
        (
            _basic_structured_prompt(extra_user_block=True),
            {},
            "This prompt uses multiple structured blocks.",
        ),
        (
            build_prompt_editor_state(
                {"artifact_type": "recipe", "system_prompt": "System"}
            ),
            {},
            "Recipes require Advanced view.",
        ),
        (
            build_prompt_editor_state(
                {
                    "id": 9,
                    "artifact_type": "prompt",
                    "prompt_format": "foreign",
                    "prompt_schema_version": 1,
                    "system_prompt": "Compatibility text",
                }
            ),
            {},
            "This prompt requires compatibility or conversion controls.",
        ),
        (
            _basic_structured_prompt(),
            {"conflict": True},
            "Resolve the version conflict in Advanced view.",
        ),
        (
            _basic_structured_prompt(),
            {"can_update_original": False},
            "This saved prompt cannot be safely updated from Basic view.",
        ),
    ],
    ids=["multi-block", "recipe", "compatibility", "conflict", "read-only"],
)
def test_prompt_basic_rejects_structure_and_safety_states(state, kwargs, expected):
    assert prompt_basic_unavailable_reason(state, **kwargs) == expected
