"""The pipeline middleware contract: declared == implemented (TASK-17600).

TASK-16965's sweep found `result_reranking` -- a middleware the shipped
`rag_pipelines.toml` declares with `enabled = true`, lists on the
`high_accuracy` pipeline (chosen *because* it promises accuracy), and whose
handler in `PipelineLoader._apply_after_middleware` is a bare `pass`.

Enumerating the whole namespace rather than that one name found **eleven**
middleware names declared by pipelines against **four** handler branches.
`result_reranking` was merely the honest one: it at least had a branch with
a `pass` in it. The other seven fell off the end of an `if/elif` and no-op'd
in total silence, with nothing in the file marking them as unbuilt.

That is the recurring species this programme keeps finding one instance of
at a time (TASK-16174's parent-inclusion knobs; `reranking_strategy`): a
surface that is declared, switched on, and documented while implementing
nothing. **This module is the guard that ends the class for this namespace**
-- it is the deliverable, more than any single deletion, and it runs in BOTH
directions:

* declared-but-unimplemented -- a pipeline naming a middleware nobody wrote;
* a branch that exists but does nothing -- the `pass`;
* implemented-but-undeclared -- a handler no pipeline can reach, which is
  the same lie told backwards;
* declared-but-undefined -- a pipeline naming a middleware with no
  `[middleware.*]` block at all, which `_apply_middleware` skips silently
  because it only runs `mid_id in self.middleware`.

The implemented set is read from the SOURCE with `ast` rather than by
importing and probing, because what is under test is which names the
`if/elif` chains actually branch on -- that is a syntactic fact, and a
runtime probe would need one crafted call per name and would still not see
a branch that exists but does nothing.
"""

import ast
import tomllib
from pathlib import Path
from typing import Dict, Set

import pytest

import tldw_chatbook
from tldw_chatbook.RAG_Search import pipeline_loader as pipeline_loader_module

#: The file the app ships (and copies into the user's config dir on first
#: load). The user's copy can drift; this is the one the project is
#: responsible for.
PIPELINES_TOML = (
    Path(tldw_chatbook.__file__).parent / "Config_Files" / "rag_pipelines.toml"
)

#: `pipeline.middleware` phase -> the handler that dispatches that phase.
_PHASE_HANDLERS = {
    "before": "_apply_before_middleware",
    "after": "_apply_after_middleware",
    "error": "_apply_error_middleware",
}


def _toml() -> dict:
    with open(PIPELINES_TOML, "rb") as f:
        return tomllib.load(f)


def _declared_by_phase() -> Dict[str, Set[str]]:
    """Every middleware name a pipeline lists, keyed by phase."""
    declared: Dict[str, Set[str]] = {phase: set() for phase in _PHASE_HANDLERS}
    for pipeline in _toml().get("pipelines", {}).values():
        for phase, names in (pipeline.get("middleware") or {}).items():
            declared.setdefault(phase, set()).update(names)
    return declared


def _handler_nodes() -> Dict[str, ast.AST]:
    """The `_apply_*_middleware` function bodies, from the source."""
    source = Path(pipeline_loader_module.__file__).read_text()
    tree = ast.parse(source)
    wanted = set(_PHASE_HANDLERS.values())
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    }


def _branch_bodies(handler: ast.AST) -> Dict[str, list]:
    """Map each name the handler branches on to that branch's body.

    Covers `middleware_id == "x"` and `middleware_id in ("x", "y")`; a new
    dispatch shape (a dict, a match statement) would show up here as an
    empty result rather than a false pass, because the declared set would
    then have no implementations at all.
    """
    bodies: Dict[str, list] = {}
    for node in ast.walk(handler):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or not isinstance(test.left, ast.Name):
            continue
        if test.left.id != "middleware_id":
            continue
        for op, comparator in zip(test.ops, test.comparators):
            if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant):
                bodies[comparator.value] = node.body
            elif isinstance(op, ast.In) and isinstance(
                comparator, (ast.Tuple, ast.List, ast.Set)
            ):
                for element in comparator.elts:
                    if isinstance(element, ast.Constant):
                        bodies[element.value] = node.body
    return bodies


def _is_a_no_op(body: list) -> bool:
    """True when a branch body does nothing: `pass`, `...`, or a bare string.

    Comments are not in the AST at all, so a branch of pure commentary
    reduces to exactly this.
    """
    return all(
        isinstance(statement, ast.Pass)
        or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
        )
        for statement in body
    )


def _implemented_by_phase() -> Dict[str, Dict[str, list]]:
    handlers = _handler_nodes()
    return {
        phase: _branch_bodies(handlers[name]) if name in handlers else {}
        for phase, name in _PHASE_HANDLERS.items()
    }


@pytest.mark.parametrize("phase", sorted(_PHASE_HANDLERS))
def test_every_declared_middleware_name_is_implemented(phase):
    """No pipeline may name a middleware that has no handler branch.

    A user picking `high_accuracy` is buying the middleware chain the file
    advertises; a name with nothing behind it spends their attention (and,
    for anything that would have called a provider, their money) on a
    promise the code never kept.
    """
    declared = _declared_by_phase().get(phase, set())
    implemented = set(_implemented_by_phase()[phase])
    missing = sorted(declared - implemented)
    assert not missing, (
        f"pipelines declare {phase}-search middleware with no implementation "
        f"in {_PHASE_HANDLERS[phase]}: {missing}. Either implement them or "
        f"remove them from {PIPELINES_TOML.name} -- a declared no-op is a "
        f"promise the app does not keep."
    )


@pytest.mark.parametrize("phase", sorted(_PHASE_HANDLERS))
def test_no_declared_middleware_is_a_bare_pass(phase):
    """A branch that exists but does nothing is not an implementation.

    This is `result_reranking`'s exact shape and the reason the task was
    filed: `enabled = true`, listed by a shipped pipeline, handled by
    `pass`. Branch PRESENCE alone would have called it implemented.
    """
    declared = _declared_by_phase().get(phase, set())
    implemented = _implemented_by_phase()[phase]
    no_ops = sorted(
        name
        for name, body in implemented.items()
        if name in declared and _is_a_no_op(body)
    )
    assert not no_ops, (
        f"{_PHASE_HANDLERS[phase]} handles these declared names with a body "
        f"that does nothing: {no_ops}. A `pass` is not an implementation."
    )


@pytest.mark.parametrize("phase", sorted(_PHASE_HANDLERS))
def test_no_middleware_implementation_is_undeclared(phase):
    """The reverse direction: a handler branch no pipeline can reach.

    Currently a GUARD, not a repair proof -- it was green before this task's
    deletions and stays green after them. It is here because the filed AC
    only anticipated this direction while the actual gap ran the other way,
    and one direction alone would have missed it.
    """
    declared = _declared_by_phase().get(phase, set())
    implemented = set(_implemented_by_phase()[phase])
    unreachable = sorted(implemented - declared)
    assert not unreachable, (
        f"{_PHASE_HANDLERS[phase]} implements middleware no pipeline lists: "
        f"{unreachable}. Dead code, or a pipeline that lost its declaration."
    )


@pytest.mark.parametrize("phase", sorted(_PHASE_HANDLERS))
def test_every_declared_middleware_name_has_a_definition_block(phase):
    """A declared name with no `[middleware.*]` block references nothing.

    `_apply_middleware` gates on `mid_id in self.middleware`, so such a name
    is not even dispatched -- it is inert one layer earlier than an
    unimplemented handler, and equally invisible.
    """
    declared = _declared_by_phase().get(phase, set())
    defined = set(_toml().get("middleware", {}))
    undefined = sorted(declared - defined)
    assert not undefined, (
        f"pipelines list {phase}-search middleware with no [middleware.*] "
        f"block in {PIPELINES_TOML.name}: {undefined}."
    )


def test_the_guard_can_see_the_names_it_is_guarding():
    """The guard's own oracle: if the parsers ever return nothing -- a moved
    file, a renamed handler, a new dispatch shape -- every assertion above
    passes vacuously. Pin that both halves are non-empty."""
    assert PIPELINES_TOML.exists(), PIPELINES_TOML
    assert _handler_nodes(), "the _apply_*_middleware handlers were not found"
    assert any(_declared_by_phase().values()), "no pipeline declares middleware"
    assert any(_implemented_by_phase().values()), "no middleware branch was parsed"


# ---------------------------------------------------------------------------
# Qodo PR-1778: the deletion has to reach users who already have a config copy
# ---------------------------------------------------------------------------


def test_unimplemented_middleware_is_dropped_from_a_user_config(tmp_path):
    """TASK-17600 deleted eight middleware promises from the BUNDLED toml, but
    `PipelineLoader` copies that file to the user's config directory on first
    run and prefers the copy forever after. Every existing installation
    therefore keeps all eight stale declarations -- and now the branch that
    at least recognised `result_reranking` is gone too.

    The fix is a RUNTIME rule rather than a migration, for the reason
    TASK-17365 chose a floor over rewriting saved profiles: a loader may
    safely refuse to honour a name it cannot implement, but it should not
    silently rewrite a file the user owns.
    """
    from tldw_chatbook.RAG_Search.pipeline_loader import PipelineLoader

    stale = tmp_path / "rag_pipelines.toml"
    stale.write_text(
        """
[pipelines.legacy]
name = "Legacy"
description = "a config copied before TASK-17600"
type = "built-in"
function = "search_plain"
enabled = true

[pipelines.legacy.middleware]
before = ["query_expansion", "code_syntax_enhancer"]
after = ["result_reranking", "citation_enhancement", "table_renderer"]
""",
        encoding="utf-8",
    )

    loader = PipelineLoader()
    loader.load_pipeline_config(stale)

    pipeline = loader.pipelines["legacy"]
    # `PipelineConfig.middleware` is a dict of phase -> names, not an object.
    before = list(pipeline.middleware.get("before") or [])
    after = list(pipeline.middleware.get("after") or [])

    assert before == ["query_expansion"], f"stale before-names survived: {before}"
    assert after == ["citation_enhancement"], f"stale after-names survived: {after}"
