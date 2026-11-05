"""Installed policy validation preserves refusals without per-literal recursion."""

import cProfile
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_chatbook.Backup_Recovery.models import SchemaPolicy


def _old_accepts(values):
    def immutable(value):
        return type(value) in (str, int) or (
            type(value) is tuple and all(immutable(item) for item in value)
        )

    return all(type(value) is tuple and immutable(value) for value in values)


_nodes = st.recursive(
    st.one_of(
        st.text(max_size=12),
        st.integers(),
        st.none(),
        st.booleans(),
        st.binary(max_size=8),
        st.lists(st.integers(), max_size=2),
    ),
    lambda children: st.lists(children, max_size=3).map(tuple),
    max_leaves=16,
)


@pytest.mark.parametrize("root", range(3))
@given(st.lists(_nodes, max_size=4).map(tuple))
def test_nested_policy_acceptance_matches_original_exact_type_contract(root, value):
    values = [(), (), ()]
    values[root] = value
    if _old_accepts(values):
        assert SchemaPolicy("fixture", *values).owner == "fixture"
    else:
        with pytest.raises(TypeError, match="^immutable_schema_policy_required$"):
            SchemaPolicy("fixture", *values)


@pytest.mark.parametrize("root", range(3))
@pytest.mark.parametrize("value", [None, True, 1, "sql", [], {}, set()])
def test_policy_roots_require_exact_tuples(root, value):
    values = [(), (), ()]
    values[root] = value
    with pytest.raises(TypeError, match="^immutable_schema_policy_required$"):
        SchemaPolicy("fixture", *values)


def test_policy_subclasses_and_type_comparison_short_circuit_stay_unchanged():
    class Text(str):
        pass

    class Number(int):
        pass

    class Sequence(tuple):
        pass

    for value in (Text("sql"), Number(1), Sequence()):
        with pytest.raises(TypeError, match="^immutable_schema_policy_required$"):
            SchemaPolicy("fixture", (value,), (), ())
    with pytest.raises(TypeError, match="^immutable_schema_policy_required$"):
        SchemaPolicy("fixture", Sequence(), (), ())

    class ScalarType(type):
        def __eq__(cls, other):
            return other is str

        __hash__ = type.__hash__

    class Scalar(metaclass=ScalarType):
        pass

    value = Scalar()
    assert _old_accepts(((value,), (), ()))
    SchemaPolicy("fixture", (value,), (), ())

    class RefusingType(type):
        def __eq__(cls, other):
            raise RuntimeError("fixture_type_comparison")

        __hash__ = type.__hash__

    class Refusing(metaclass=RefusingType):
        pass

    value = Refusing()
    with pytest.raises(TypeError, match="^immutable_schema_policy_required$"):
        SchemaPolicy("fixture", (None, value), (value,), (value,))
    with pytest.raises(RuntimeError, match="^fixture_type_comparison$"):
        SchemaPolicy("fixture", (value,), (), ())


def test_installed_core_policy_work_follows_containers_not_sql_literals():
    from tldw_chatbook.DB.recovery_core import core_adapters

    adapter = next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")
    policy = adapter.schema_policy()
    pending = [policy.versions, policy.schema_sql, policy.migration_steps]
    containers = 0
    while pending:
        item = pending.pop()
        if type(item) is tuple:
            containers += 1
            pending.extend(item)
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        observed = replace(policy)
    finally:
        profiler.disable()
    calls = sum(
        entry.callcount
        for entry in profiler.getstats()
        if not isinstance(entry.code, str)
        and entry.code.co_filename == SchemaPolicy.__post_init__.__code__.co_filename
        and entry.code.co_name == "immutable"
    )
    assert observed == policy
    assert calls <= containers, (calls, containers)
