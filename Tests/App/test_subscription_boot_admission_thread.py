"""Subscription backfill startup must leave guarded resolution to its worker."""

import asyncio
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tldw_chatbook.app import TldwCli

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [False, True])
async def test_backfill_starter_leaves_resolution_and_fresh_refusal_to_worker(
    monkeypatch, tmp_path, allowed
):
    from tldw_chatbook import app as app_module, app_feature_glue
    from tldw_chatbook.Backup_Recovery import activation

    origin = threading.get_ident()
    steps = []
    path = tmp_path / "subscriptions.db"

    def resolve():
        steps.append(("resolve", threading.get_ident()))
        assert threading.get_ident() != origin, "guarded path lookup on UI loop"
        return path

    @contextmanager
    def execution_scope(names, selected):
        assert names == ("db.subscriptions",) and selected == path
        steps.append(("admit", threading.get_ident()))
        yield allowed

    def body(_host):
        steps.append(("body", threading.get_ident()))

    queued = []
    token = object()

    def run_worker(operation, **kwargs):
        assert kwargs["thread"] and kwargs["exclusive"]
        assert kwargs["name"] == "_backfill_subscription_items_fts"
        assert kwargs["group"] == "subscriptions-fts-backfill"
        queued.append(operation)
        return token

    host = SimpleNamespace(run_worker=run_worker)
    host._backfill_subscription_items_fts = (
        lambda: TldwCli._backfill_subscription_items_fts(host)
    )
    monkeypatch.setattr(app_module, "get_subscriptions_db_path", resolve, raising=False)
    monkeypatch.setattr(app_feature_glue, "get_subscriptions_db_path", resolve)
    monkeypatch.setattr(activation, "execution_scope", execution_scope)
    monkeypatch.setattr(TldwCli, "_backfill_subscription_items_fts_owned", body)
    assert TldwCli.boot_worker_starters(host)["subscriptions_fts_backfill"]() is token
    assert steps == [] and len(queued) == 1
    await asyncio.to_thread(queued[0])
    assert [step for step, _ in steps] == ["resolve", "admit"] + (
        ["body"] if allowed else []
    )
    assert all(thread != origin for _, thread in steps)
