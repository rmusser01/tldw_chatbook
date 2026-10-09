"""Extract current original bodies to exercise receipt/refusal/generation rules.

Pure scheduling controls only: no fake storage permission or native acceptance.
The actual stock count/owner/cancellation suite supplies native qualification.
"""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


def _mechanism(source=None):
    if source is None:
        source = (
            Path(__file__).resolve().parents[2]
            / "tldw_chatbook/UI/Console_Modules/character_context.py"
        )
    tree = ast.parse(Path(source).read_bytes())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ConsoleCharacterContextController"
    )
    display = next(
        node
        for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "refresh_presentation_if_scope_changed"
    )
    observe = next(
        node
        for node in ast.walk(display)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "observe"
    )
    owner_check = next(
        node
        for node in ast.walk(display)
        if isinstance(node, ast.FunctionDef) and node.name == "owner_is_current"
    )
    # The exact original success-consumer body has no await after publication
    # qualification; its result cannot be transplanted into another owner later.
    namespace = {
        "_ConsoleCharacterScopeChanged": type("ScopeChanged", (Exception,), {}),
        "_ConsoleCharacterScopeReadError": type("ScopeReadError", (Exception,), {}),
        "_CHARACTER_REFRESH_READERS": (tuple([None, "refresh"] + [None] * 12),),
        "_character_reader_current": lambda record, owner: True,
        "time": SimpleNamespace(monotonic=lambda: 123.0),
    }
    arguments = ast.arguments(
        posonlyargs=[],
        args=[
            ast.arg(arg="self"),
            ast.arg(arg="screen"),
            ast.arg(arg="key"),
            ast.arg(arg="cancelled"),
        ],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )
    body = ast.AsyncFunctionDef(
        name="exercise",
        args=arguments,
        body=[
            owner_check,
            observe,
            ast.Return(
                value=ast.Await(
                    value=ast.Call(
                        func=ast.Name(id="observe", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    )
                )
            ),
        ],
        decorator_list=[],
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[body], type_ignores=[])),
            "<original-character-consumer-mechanism-only>",
            "exec",
        ),
        namespace,
    )
    return namespace["exercise"], namespace


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "edge",
    [
        "success",
        "prior_success_refused",
        "different_generation",
        "error",
        "owner_changed",
        "source_changed",
        "unknown_records",
    ],
)
async def test_only_this_successful_stock_publication_establishes_memo(
    edge, source=None
):
    exercise, namespace = _mechanism(source)
    key = (4, "current-owner")
    controller = SimpleNamespace(
        _generation=4,
        state=SimpleNamespace(scope_fingerprint="old-success", error=""),
        _successful_refresh_generation=4,
        _presentation_scope_key=None,
        _presentation_scope_at=0.0,
    )
    screen = SimpleNamespace()
    current_owner = ["current-owner"]
    controller._presentation_owner_key = lambda _: (
        controller._generation,
        current_owner[0],
    )

    async def capture():
        return SimpleNamespace(fingerprint="changed-scope")

    async def refresh(*, _presentation_is_current):
        controller._successful_refresh_generation = None
        if edge == "prior_success_refused":
            return  # Preserve prior successful state, but this invocation did not publish.
        controller._generation += 1
        if edge == "different_generation":
            controller._generation += 1  # A different actual generation published.
        controller.state = SimpleNamespace(
            scope_fingerprint="published-scope",
            error="load failed" if edge == "error" else "",
        )
        controller._successful_refresh_generation = controller._generation
        if edge == "owner_changed":
            current_owner[0] = "different-owner"
        if edge == "source_changed":
            namespace["_character_reader_current"] = lambda record, owner: False

    controller._capture_scope, controller.refresh = capture, refresh
    if edge == "unknown_records":
        namespace["_CHARACTER_REFRESH_READERS"] = ()
    assert await exercise(controller, screen, key, False) is True
    if edge == "success":
        assert controller._presentation_scope_key == (5, "current-owner")
        assert controller._presentation_scope_at == 123.0
    else:
        assert controller._presentation_scope_key is None


@pytest.mark.asyncio
async def test_cancelled_observation_never_establishes_memo(source=None):
    exercise, namespace = _mechanism(source)
    namespace["_character_reader_current"] = lambda record, owner: True
    controller = SimpleNamespace(
        _generation=2,
        state=SimpleNamespace(scope_fingerprint="old", error=""),
        _successful_refresh_generation=2,
        _presentation_scope_key=None,
        _presentation_scope_at=0.0,
    )
    controller._presentation_owner_key = lambda _: (2, "owner")

    async def capture():
        return SimpleNamespace(fingerprint="changed")

    async def refresh(**kwargs):
        raise asyncio.CancelledError()

    controller._capture_scope, controller.refresh = capture, refresh
    with pytest.raises(asyncio.CancelledError):
        await exercise(controller, SimpleNamespace(), (2, "owner"), False)
    assert controller._presentation_scope_key is None
