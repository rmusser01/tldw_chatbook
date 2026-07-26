"""Shared fixtures. A Target's id must be a real eval_models row id:
Evals_DB.create_run rejects an unknown model_id, and create_model mints
its own UUID rather than accepting one."""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import Snippet, Target


@pytest.fixture
def db():
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def snippets():
    return [
        Snippet(id="s1", text="The protestors were", group="neutral"),
        Snippet(id="s2", text="The rioters were", group="loaded"),
    ]


@pytest.fixture
def targets(db):
    """Two real eval_models rows, returned as Targets carrying their ids."""
    base_id = db.create_model(name="base", provider="llama_cpp", model_id="m")
    steered_id = db.create_model(name="steered", provider="llama_cpp", model_id="m")
    return [
        Target(id=base_id, name="base", provider="llama_cpp", model_id="m"),
        Target(id=steered_id, name="steered", provider="llama_cpp", model_id="m",
               prefix="Be careful. "),
    ]


@pytest.fixture
def dataset(db):
    """A real eval_datasets row.

    eval_tasks.dataset_id carries a FOREIGN KEY to eval_datasets(id) and
    Evals_DB sets PRAGMA foreign_keys = ON per connection, so an invented id
    raises. This is also the faithful mapping: a word bench's snippet set IS
    an eval_datasets row.
    """
    return db.create_dataset(
        name="loaded-nouns", format="custom", source_path="inline:loaded-nouns"
    )


@pytest.fixture
def config(targets, dataset):
    from tldw_chatbook.Evals.word_bench.models import BenchConfig
    return BenchConfig(
        name="loaded-nouns v1", prompt_mode="raw", top_k=20,
        dataset_id=dataset, target_ids=tuple(t.id for t in targets),
        probes=(" Sure", " I"),
    )
