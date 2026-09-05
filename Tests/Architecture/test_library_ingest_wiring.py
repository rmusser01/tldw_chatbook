"""Ingest extraction series: state object and controller are screen-wired.

Wave-5 Task 1 (state PR -- ingest series 1/3), Task 2 (controller PR --
ingest series 2/3), and Task 3 (cleanup PR -- ingest series 3/3; recipe:
backlog/docs/library-decomposition-recipe.md; skills/export/collections/
search+RAG series precedent: Tests/Architecture/test_library_skills_wiring.py
/ test_library_export_wiring.py / etc.).

Task 1's ``test_state_object_fields_match_the_shim_surface`` (every
``LibraryIngestState`` field <-> a matching generated property shim on
``LibraryScreen``) is GONE as of Task 3: the screen's generated shim block
was deleted wholesale in cleanup (``self._ingest_state`` is a real
``LibraryIngestState`` instance now, not a shimmed screen attribute), so
there is nothing left on ``LibraryScreen`` for that assertion to check --
the skills/export/collections/search+RAG series' own Task-3/4-shaped
cleanup precedent. ``test_ingest_controller_exposes_every_state_field``
below already covers the equivalent job on the controller side and needed
no change.

Task 2 added the full-cluster ownership/same-name-delegator-forwarding
checks (``_INGEST_CLUSTER_METHOD_NAMES``, 56 names). See
``library_ingest_controller.py``'s own module docstring for the full
78-candidate derivation and the 22 exclusions (4 ``@work`` framework-
decorator hazard, 3 module-globals-coupling, 9 unbound-fake-self/
``object.__new__``-bypass, 6 instance-attribute-monkeypatch).

Task 2 fix round 1 (post-review): ``_resolve_ingest_source`` moved back
to the exclusion list (module-globals coupling on ``validate_path_
simple``/``validate_url``, found by the coordinator-mandated mechanical
module-globals census, not the original battery -- see ``library_ingest_
controller.py``'s own module docstring for the full incident, including
the existing-file probe that confirmed it).

Task 3 adds the ``_INGEST_CLUSTER_SCREEN_DELEGATOR_PRUNED`` skip/absence-
assertion pair to ``test_screen_delegates_ingest_handlers`` (6 names, incl.
the cluster's one staticmethod -- see that frozenset's own docstring for
the repo-wide zero-external-reference census) and the equivalent absence
assertion for the now-pruned staticmethod delegator in
``test_ingest_cluster_staticmethods_forward_to_the_controller_class``,
mirroring ``_SKILLS_CLUSTER_SCREEN_DELEGATOR_PRUNED`` in the skills wiring
test.
"""
from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from tldw_chatbook.UI.Library_Modules.library_ingest_state import (
    LibraryIngestState,
)


#: Every method Task 2 moved into `LibraryIngestController`, under its
#: original `LibraryScreen` name. Derived from a full `ast` census of every
#: `LibraryScreen` method whose name contains "ingest" (78 raw matches, 78
#: unique -- matching Task 1's own census), minus 4 `@work` framework-
#: decorator-hazard exclusions, 3 module-globals-coupling exclusions
#: (`_remember_library_ingest_location`, `_load_library_ingest_options_
#: from_config`, and `_resolve_ingest_source` -- the last one moved back
#: here in fix round 1, found by the mechanical module-globals census, not
#: the original battery), 9 unbound-fake-self/`object.__new__`-bypass
#: exclusions, and 6 instance-attribute-monkeypatch exclusions
#: (`_build_library_ingest_state`, `_library_ingest_job_by_id`,
#: `_notify_library_ingest_warning`, `_refresh_library_ingest_canvas_
#: preserving_context`, `_update_library_ingest_dynamic_regions`,
#: `_update_library_ingest_gate`) -- NOT a prefix/substring shortcut. See
#: `library_ingest_controller.py`'s module docstring for the full per-name
#: reasoning behind every exclusion.
_INGEST_CLUSTER_METHOD_NAMES: tuple[str, ...] = (
    "_adopt_library_ingest_path",
    "_apply_library_ingest_backend_save",
    "_apply_library_ingest_preflight_result",
    "_authoritative_library_ingest_consent_is_current",
    "_cancel_library_ingest_preflight",
    "_current_library_ingest_start_consent",
    "_disarm_library_ingest_retry_confirm",
    "_disarm_library_ingest_start_confirm",
    "_focus_library_ingest_path",
    "_handle_library_ingest_progress_changed",
    "_handle_library_ingest_registry_changed",
    "_ingest_job_id_from_button",
    "_invalidate_library_ingest_preflight",
    "_library_ingest_registry",
    "_library_ingest_restage_discards_work",
    "_library_ingest_shortcuts_for_current_state",
    "_on_ingest_job_details",
    "_on_library_ingest_top_button",
    "_pause_library_ingest_transient_ui",
    "_reset_library_ingest_transient_state",
    "_restage_library_ingest_last_submission",
    "_restore_library_ingest_canvas_context",
    "_scroll_library_ingest_queue_into_view",
    "_set_library_ingest_panels_collapsed",
    "_submit_library_ingest_form",
    "_sync_library_ingest_rail_for_width",
    "_sync_library_ingest_rail_from_shell",
    "_trigger_library_ingest_preflight",
    "_update_library_ingest_fold_hint",
    "_update_library_ingest_group_receipt",
    "_update_library_ingest_retry_label",
    "action_library_ingest_back",
    "action_library_ingest_retry_last",
    "handle_library_ingest_author_changed",
    "handle_library_ingest_browse",
    "handle_library_ingest_cancel",
    "handle_library_ingest_choose_gguf",
    "handle_library_ingest_clear_finished",
    "handle_library_ingest_clear_path",
    "handle_library_ingest_collapse_all",
    "handle_library_ingest_dismiss",
    "handle_library_ingest_expand_all",
    "handle_library_ingest_force_stop",
    "handle_library_ingest_keywords_changed",
    "handle_library_ingest_open",
    "handle_library_ingest_path_blurred",
    "handle_library_ingest_path_changed",
    "handle_library_ingest_path_submitted",
    "handle_library_ingest_retry",
    "handle_library_ingest_retry_faster_whisper",
    "handle_library_ingest_retry_last",
    "handle_library_ingest_start",
    "handle_library_ingest_title_changed",
    "handle_library_ingest_view_on_server",
    "sync_library_ingest_tooling_detail_expanded",
    "sync_library_ingest_type_group_expanded",
)

#: The 1 name above that is a `@staticmethod` on `LibraryScreen`. Its
#: delegator forwards straight to the module-level `LibraryIngestController`
#: CLASS (per the skills/export/collections/search+RAG wiring tests'
#: "static-method delegator pattern" precedent), not through
#: `self._ingest_controller`.
_INGEST_CLUSTER_STATICMETHOD_NAMES: frozenset[str] = frozenset(
    {
        "_ingest_job_id_from_button",
    }
)

#: Task 3's own repo-wide zero-external-reference census (`tldw_chatbook/`
#: + every `Tests/` root, excluding `library_ingest_controller.py` itself
#: and each name's own one-line screen delegator body): of the 56 moved
#: names, 25 `@on` handlers + 2 `action_*` methods KEEP unconditionally
#: per the recipe's own transform whitelist (`@on`/`action_*` dispatch is
#: never visible to a grep-based census); of the remaining 29 (28 plain +
#: 1 staticmethod), 23 were kept for a genuine external caller (an
#: excluded, still-screen-resident method calling `self.<name>()`, or a
#: test calling/patching the screen delegator directly) and 6 had none --
#: this 6-of-29 (~21%) prune fraction sits within the export/skills/
#: search+RAG/collections/conversations series' own recorded range
#: (~5%-~30%).
_INGEST_CLUSTER_SCREEN_DELEGATOR_PRUNED: frozenset[str] = frozenset(
    {
        "_adopt_library_ingest_path",
        "_ingest_job_id_from_button",
        "_library_ingest_restage_discards_work",
        "_restage_library_ingest_last_submission",
        "_set_library_ingest_panels_collapsed",
        "_update_library_ingest_retry_label",
    }
)


@pytest.mark.unit
def test_ingest_cluster_method_names_are_genuinely_ingest_named() -> None:
    """Guards the hand-kept cluster list against drift with the census: every
    name must contain "ingest" (case-insensitive) -- a typo here would
    silently test the wrong surface.
    """
    not_ingest_named = [
        n for n in _INGEST_CLUSTER_METHOD_NAMES if "ingest" not in n.lower()
    ]
    assert not not_ingest_named, f"non-ingest-named cluster entries: {not_ingest_named!r}"
    assert len(_INGEST_CLUSTER_METHOD_NAMES) == 56, (
        f"expected 56 moved names, got {len(_INGEST_CLUSTER_METHOD_NAMES)}"
    )


@pytest.mark.unit
def test_ingest_controller_owns_its_cluster() -> None:
    """Every one of the 56 moved names is a callable on the controller.

    Covers the whole cluster, not a hand-picked sample -- mirrors
    `test_skills_controller_owns_its_cluster`.
    """
    from tldw_chatbook.UI.Library_Modules.library_ingest_controller import (
        LibraryIngestController,
    )

    missing = [
        name
        for name in _INGEST_CLUSTER_METHOD_NAMES
        if not callable(getattr(LibraryIngestController, name, None))
    ]
    assert not missing, f"LibraryIngestController is missing: {missing!r}"


@pytest.mark.unit
def test_screen_delegates_ingest_handlers() -> None:
    """Every one of the 56 moved names is a one-line screen delegator that
    forwards to the SAME-NAMED controller method (or, for the 1
    staticmethod, to the module-level controller CLASS) -- unless Task 3
    pruned it.

    Mirrors `test_screen_delegates_skills_handlers`: a same-name forwarding
    check, not a loose "the controller is referenced somewhere" substring
    check. Skips `_INGEST_CLUSTER_SCREEN_DELEGATOR_PRUNED` (Task 3 deleted
    those 6 screen delegators as dead weight -- zero external references)
    and instead asserts each such name is genuinely ABSENT from
    `LibraryScreen`, so a future accidental re-add would fail loudly here
    rather than silently reintroducing dead code.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_delegators = []
    for name in _INGEST_CLUSTER_METHOD_NAMES:
        if name in _INGEST_CLUSTER_SCREEN_DELEGATOR_PRUNED:
            assert getattr(LibraryScreen, name, None) is None, (
                f"{name!r} was pruned from the screen (task 3) but is back -- "
                "either wire it as a delegator again or drop it from "
                "_INGEST_CLUSTER_SCREEN_DELEGATOR_PRUNED"
            )
            continue
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_delegators.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        escaped = re.escape(name)
        if not re.search(
            rf"_ingest_controller\.{escaped}\(", src
        ) and not re.search(rf"LibraryIngestController\.{escaped}\(", src):
            not_delegators.append(name)
    assert not not_delegators, f"not delegators yet: {not_delegators!r}"


@pytest.mark.unit
def test_ingest_cluster_staticmethods_forward_to_the_controller_class() -> None:
    """The 1 staticmethod name in the cluster forwarded to the CLASS.

    Task 3 pruned it (`_ingest_job_id_from_button` had zero external
    references -- see `_INGEST_CLUSTER_SCREEN_DELEGATOR_PRUNED`), so this
    now asserts its genuine absence instead of a class-forwarding shape.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_class_forwarding = []
    for name in _INGEST_CLUSTER_STATICMETHOD_NAMES:
        if name in _INGEST_CLUSTER_SCREEN_DELEGATOR_PRUNED:
            assert getattr(LibraryScreen, name, None) is None, (
                f"{name!r} was pruned from the screen (task 3) but is back"
            )
            continue
        src = inspect.getsource(getattr(LibraryScreen, name))
        if not re.search(rf"LibraryIngestController\.{re.escape(name)}\(", src):
            not_class_forwarding.append(name)
    assert not not_class_forwarding, (
        f"expected class-forwarding delegators: {not_class_forwarding!r}"
    )


@pytest.mark.unit
def test_ingest_controller_exposes_every_state_field() -> None:
    """The controller's generated shim loop covers every state field.

    Mirrors `test_skills_controller_exposes_every_state_field`, minus the
    prefix-mapping indirection (Ingest uses a single flat `_library_ingest_`
    prefix -- task 1's own finding, no plural variant needed).
    """
    from tldw_chatbook.UI.Library_Modules.library_ingest_controller import (
        LibraryIngestController,
    )

    field_names = {f.name for f in dataclasses.fields(LibraryIngestState)}
    assert field_names, "state object is empty"
    missing = []
    for name in field_names:
        shim_attr = "_library_ingest_" + name
        if not isinstance(getattr(LibraryIngestController, shim_attr, None), property):
            missing.append(shim_attr)
    assert not missing, (
        f"no ingest controller shim property found for state field(s): {missing!r}"
    )
