"""A native review actor retains the original conflicting worktree claim."""

from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile


def test_held_native_review_child_blocks_workflow_claim_until_physical_return(tmp_path):
    _run(
        tmp_path,
        "review-held",
        "native",
        script=Path(__file__).with_name("workflow_review_probe.py").read_text(),
        timeout=90,
    )


def test_ordinary_native_child_returns_its_original_source_before_finishing(tmp_path):
    _run(
        tmp_path,
        "ordinary-held",
        "native",
        script=Path(__file__).with_name("workflow_review_probe.py").read_text(),
        timeout=90,
    )


def test_ordinary_child_scope_return_failure_retains_same_original_custody(tmp_path):
    with pytest.raises(
        AssertionError, match="ORIGINAL_CHILD_SCOPE_RETURN_UNCERTAIN_CUSTODY_RETAINED"
    ):
        _run(
            tmp_path,
            "ordinary-scope-error",
            "retained",
            script=Path(__file__).with_name("workflow_review_probe.py").read_text(),
            timeout=90,
        )


def _capture(text, boundary=None):
    from types import MethodType, SimpleNamespace
    from tldw_chatbook.Chat.console_ordinary_tree import (
        _OrdinaryTurn,
        _capture_review_input,
    )

    # Pure metadata projection: no runtime owner, provider or permission is minted.
    tree = SimpleNamespace(
        local_source_values=(),
        local_function_values=(),
        raw_projection=None,
        check=lambda: None,
    )
    tree.check_local_sources = MethodType(_OrdinaryTurn.check_local_sources, tree)
    _capture_review_input(tree, text, boundary)
    return tree


@pytest.mark.parametrize(
    "text,possible",
    [
        ("Read input.txt", False),
        ("Explain how code review works", False),
        ("If the tests pass, review my changes", False),
        ('"Review my changes"', False),
        ("Review my changes", True),
        ("$multi-model-review Review my changes", True),
        ("yes", True),
    ],
)
def test_review_maximum_uses_native_intent_without_widening_other_input(text, possible):
    assert _capture(text).review_possible is possible


@pytest.mark.parametrize("late", [False, True])
def test_replaced_native_review_classifier_is_refused_without_invocation(
    monkeypatch, late
):
    from tldw_chatbook.Chat import console_code_review as review
    from tldw_chatbook.Chat.console_ordinary_tree import check_ordinary_review_input
    from types import SimpleNamespace

    tree = _capture("Review my changes") if late else None
    calls = []
    monkeypatch.setattr(review, "_explicit_stages", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="ordinary .*source|ordinary .*maximum"):
        if late:
            check_ordinary_review_input(
                tree, SimpleNamespace(content="Review my changes"), None
            )
        else:
            _capture("Review my changes")
    assert not calls


@pytest.mark.parametrize("changed", ["text", "boundary"])
def test_review_receiver_refuses_changed_input_before_composition(changed):
    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_ordinary_tree import check_ordinary_review_input

    boundary = object()
    tree = _capture("Review my changes", boundary)
    message = SimpleNamespace(
        content="Review my changes"
        if changed == "boundary"
        else "Review another folder"
    )
    authorization = SimpleNamespace(
        review_boundary=object() if changed == "boundary" else boundary
    )
    with pytest.raises(RuntimeError, match="ordinary review input changed"):
        check_ordinary_review_input(tree, message, authorization)


@pytest.mark.parametrize("late", [False, True])
def test_changed_native_review_provider_is_refused_before_composition(
    monkeypatch, late
):
    from types import SimpleNamespace
    from tldw_chatbook.Agents.code_review_tool_provider import CodeReviewToolProvider
    from tldw_chatbook.Chat.console_ordinary_tree import check_ordinary_review_input

    tree = _capture("Review my changes") if late else None
    calls = []
    monkeypatch.setattr(
        CodeReviewToolProvider, "list_catalog", lambda self: calls.append(self)
    )
    with pytest.raises(RuntimeError, match="ordinary .*provider|ordinary .*maximum"):
        if late:
            check_ordinary_review_input(
                tree, SimpleNamespace(content="Review my changes"), None
            )
        else:
            _capture("Review my changes")
    assert not calls


@pytest.mark.parametrize(
    "name", ["_capture_review_input", "check_ordinary_review_input"]
)
@pytest.mark.parametrize("late", [False, True])
def test_changed_review_restriction_helper_is_refused(monkeypatch, name, late):
    from types import SimpleNamespace
    from tldw_chatbook.Chat import console_ordinary_tree as ordinary

    capture, check = (
        ordinary._capture_review_input,
        ordinary.check_ordinary_review_input,
    )
    tree = _capture("Review my changes") if late else SimpleNamespace()
    calls = []
    monkeypatch.setattr(ordinary, name, lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="ordinary .*source|ordinary .*maximum"):
        if late:
            check(tree, SimpleNamespace(content="Review my changes"), None)
        else:
            capture(tree, "Review my changes", None)
    assert not calls


@pytest.mark.parametrize("change", ["replacement", "body"])
def test_changed_extracted_review_receiver_refuses_before_source_activation(
    monkeypatch, change
):
    from tldw_chatbook.Chat import console_review_delivery as delivery
    from tldw_chatbook.Chat.console_agent_sources import _AgentSources

    original = delivery.compose_turn_review_provider
    calls = []

    async def replacement(*args, **kwargs):
        calls.append(args)

    if change == "replacement":
        monkeypatch.setattr(delivery, "compose_turn_review_provider", replacement)
    else:
        # The replacement has no closure, like the actual extracted function.
        async def changed(*args, **kwargs):
            raise AssertionError("changed extracted receiver invoked")

        monkeypatch.setattr(original, "__code__", changed.__code__)
    with pytest.raises(
        PermissionError, match="original extracted native receiver changed"
    ):
        _AgentSources._check_original_bindings(object.__new__(_AgentSources))
    assert not calls


@pytest.mark.parametrize("attached", [False, True])
def test_ordinary_binding_preserves_only_its_exact_original_member(attached):
    from threading import Lock
    from types import SimpleNamespace
    from tldw_chatbook.Agents.execution_capacity import (
        RuntimeCapacity,
        WorkOrigin,
        current_execution_owner,
    )
    from tldw_chatbook.Chat.console_ordinary_tree import bind_ordinary_tree

    capacity = RuntimeCapacity()
    primary = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="primary"
    )
    child = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="child", child=True
    )
    tree = SimpleNamespace(
        check=lambda: None,
        lock=Lock(),
        owner=primary,
        members={primary: object(), **({child: object()} if attached else {})},
    )
    try:
        with child.activate():
            with bind_ordinary_tree(tree):
                assert current_execution_owner() is (child if attached else primary)
            assert current_execution_owner() is child
    finally:
        child.finish_root()
        primary.finish_root()
    assert not capacity.snapshot().executions


def test_ordinary_binding_refuses_replaced_owner_getter_without_invocation(monkeypatch):
    from types import SimpleNamespace
    from tldw_chatbook.Agents import execution_capacity
    from tldw_chatbook.Chat.console_ordinary_tree import bind_ordinary_tree

    calls = []
    monkeypatch.setattr(
        execution_capacity, "current_execution_owner", lambda: calls.append(True)
    )
    with pytest.raises(
        RuntimeError, match="original ordinary execution owner source changed"
    ):
        with bind_ordinary_tree(SimpleNamespace(check=lambda: None)):
            raise AssertionError("changed getter admitted")
    assert not calls
