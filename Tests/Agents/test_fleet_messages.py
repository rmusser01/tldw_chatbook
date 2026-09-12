"""Real queue/capability regressions for ADR-136 progress reports."""

import copy
import dataclasses
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_chatbook.Agents.fleet_messages import (
    MessageError,
    MessageIdentity,
    MessageStore,
)


def identity(index=0, chain="chain-a"):
    return MessageIdentity(f"h-{index}", f"r-{index}", "parent", chain, "reader")


def setup_queue():
    store = MessageStore()
    inbox = store.open_inbox("conversation-a")
    sender = inbox.sender(identity())
    reader = inbox.reader("primary", chain_id="chain-a", automatic=True)
    return store, inbox, sender, reader


def assert_refusal(code, operation):
    with pytest.raises(MessageError) as exc:
        operation()
    assert exc.value.code == code
    assert str(exc.value) == code


def test_real_consumer_collects_progress():
    store, inbox, sender, reader = setup_queue()
    sender.send("Cursor pagination is required.")
    batch = reader.collect()
    assert batch.collected_count == 1
    assert "Cursor pagination is required." in batch.content
    assert batch.remaining_count == 0
    assert store.pending_counts() == {}
    assert inbox.snapshot() == ()


def test_fifo_whole_envelopes_and_detached_immutable_identity():
    _, inbox, sender, reader = setup_queue()
    ids = [sender.send(f"report {n}") for n in range(6)]
    snapshot = inbox.snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot[0].identity.chain_id = "forged"
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot[0].body = "forged"
    batch = reader.collect()
    payload = json.loads(batch.content)
    assert [m["body"] for m in payload["messages"]] == [f"report {n}" for n in range(4)]
    assert [m["message_id"] for m in payload["messages"]] == ids[:4]
    assert payload["messages"][0] == {
        "message_id": ids[0],
        "handle_id": "h-0",
        "run_id": "r-0",
        "parent_run_id": "parent",
        "chain_id": "chain-a",
        "agent": "reader",
        "body": "report 0",
    }
    assert batch.collected_count == 4
    assert payload["remaining"] == batch.remaining_count == 2
    assert len(snapshot) == 6


@pytest.mark.parametrize(
    "body,code",
    [
        (None, "invalid_message"),
        (2, "invalid_message"),
        ({"message": "a"}, "invalid_message"),
        ("", "invalid_message"),
        (" \n\t", "invalid_message"),
        ("bad\x00", "invalid_message"),
        ("bad\x1b", "invalid_message"),
        ("bad\x7f", "invalid_message"),
        ("bad\ud800", "invalid_message"),
        ("a" * 2001, "message_too_large"),
        ('"' * 2000, "message_too_large"),
    ],
)
def test_invalid_or_oversized_report_refuses_without_mutation(body, code):
    store, inbox, sender, _ = setup_queue()
    assert_refusal(code, lambda: sender.send(body))
    assert inbox.snapshot() == ()
    assert store.pending_counts() == {}
    sender.send("still available")


def test_unicode_2000_characters_and_escaped_text_survive_exactly():
    _, _, sender, reader = setup_queue()
    body = "🦉" * 2000
    sender.send(body)
    sender.send('tabs\tlines\nreturn\rquotes" slash\\')
    batch = reader.collect()
    assert len(batch.content) <= 8000
    assert [m["body"] for m in json.loads(batch.content)["messages"]] == [
        body,
        'tabs\tlines\nreturn\rquotes" slash\\',
    ]


def test_child_share_refuses_ninth_without_starving_second_child():
    _, inbox, sender, reader = setup_queue()
    for _ in range(8):
        sender.send("a" * 2000)
    assert_refusal("queue_full", lambda: sender.send("ninth"))
    inbox.sender(identity(1)).send("other child")
    assert reader.pending_count() == 9


@pytest.mark.parametrize("removal", ["collect", "discard"])
def test_lifetime_exhaustion_survives_drain_or_discard(removal):
    _, inbox, sender, reader = setup_queue()
    for _ in range(32):
        message_id = sender.send("a" * 2000)
        if removal == "collect":
            reader.collect()
        else:
            assert inbox.discard([message_id]) == 1
    assert_refusal("sender_limit", lambda: sender.send("one more"))
    assert reader.pending_count() == 0
    inbox.sender(identity(1)).send("another run")


def test_automatic_chain_filter_skips_unknown_and_foreign_fifo():
    store = MessageStore()
    inbox = store.open_inbox("a")
    for n, chain in enumerate([None, "other", "chain", "other", "chain"]):
        inbox.sender(identity(n, chain)).send(str(n))
    assert_refusal(
        "unavailable", lambda: inbox.reader("p", chain_id=None, automatic=True)
    )
    reader = inbox.reader("p", chain_id="chain", automatic=True)
    assert reader.pending_count() == 2
    batch = reader.collect()
    assert [m["body"] for m in json.loads(batch.content)["messages"]] == ["2", "4"]
    assert batch.remaining_count == 0
    assert json.loads(reader.collect().content) == {
        "status": "collected",
        "messages": [],
        "remaining": 0,
    }
    assert store.pending_counts() == {"a": 3}
    reader.close()
    manual = inbox.reader("later", chain_id=None, automatic=False)
    assert [m["body"] for m in json.loads(manual.collect().content)["messages"]] == [
        "0",
        "1",
        "3",
    ]


def test_reader_exclusivity_and_copied_capabilities_cannot_operate():
    _, inbox, sender, reader = setup_queue()
    sender.send("safe")
    assert_refusal(
        "reader_busy",
        lambda: inbox.reader("primary", chain_id="chain-a", automatic=True),
    )
    assert_refusal("unavailable", lambda: copy.copy(reader).collect())
    assert_refusal("unavailable", lambda: copy.copy(sender).send("forged"))
    reader.close()
    next_reader = inbox.reader("primary", chain_id="chain-a", automatic=True)
    reader.close()
    assert_refusal("unavailable", reader.collect)
    assert next_reader.collect().collected_count == 1


def test_close_replacement_and_noncreating_lookup_never_resurrect_capability():
    store, inbox, sender, reader = setup_queue()
    sender.send("old")
    assert store.get_inbox("missing") is None
    assert store.get_inbox("conversation-a") is inbox
    store.close_inbox("conversation-a")
    replacement = store.open_inbox("conversation-a")
    replacement.sender(identity()).send("new")
    assert_refusal("unavailable", lambda: sender.send("late"))
    assert_refusal("unavailable", reader.collect)
    assert_refusal("unavailable", inbox.snapshot)
    assert_refusal("unavailable", lambda: inbox.discard(["old"]))
    sender.close()
    reader.close()
    assert store.pending_counts() == {"conversation-a": 1}
    assert replacement.snapshot()[0].body == "new"
    store.close()
    store.close()
    assert store.pending_counts() == {}
    assert store.get_inbox("conversation-a") is None
    assert_refusal("unavailable", lambda: store.open_inbox("conversation-a"))
    assert_refusal("unavailable", replacement.snapshot)


def test_sender_revocation_preserves_reports_and_other_senders():
    _, inbox, sender, reader = setup_queue()
    sender.send("survives")
    second = inbox.sender(identity(1))
    inbox.revoke_sender("h-0")
    assert_refusal("unavailable", lambda: sender.send("late"))
    sender.close()
    second.send("live")
    assert reader.collect().collected_count == 2


def test_exact_result_cap_keeps_first_report_on_refusal():
    _, inbox, sender, reader = setup_queue()
    sender.send("first\t\n" + "🦉" * 50)
    envelope = dataclasses.asdict(inbox.snapshot()[0])
    source = envelope.pop("identity")
    expected = json.dumps(
        {"status": "collected", "messages": [{**envelope, **source}], "remaining": 0},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    assert_refusal("result_limit_too_small", lambda: reader.collect(len(expected) - 1))
    assert reader.pending_count() == 1
    assert len(reader.collect(len(expected)).content) == len(expected)


def test_result_cap_reserves_accurate_remaining_and_never_truncates():
    _, _, sender, reader = setup_queue()
    for n in range(8):
        sender.send(str(n) * 2000)
    batch = reader.collect(2500)
    assert len(batch.content) <= 2500
    assert batch.collected_count == 1
    assert batch.remaining_count == json.loads(batch.content)["remaining"] == 7
    assert json.loads(batch.content)["messages"][0]["body"] == "0" * 2000
    assert reader.pending_count() == 7


@pytest.mark.parametrize("cap", [0, -1, 100_000])
def test_nonpositive_or_large_result_cap_uses_8000_ceiling(cap):
    _, _, sender, reader = setup_queue()
    for _ in range(4):
        sender.send("a" * 2000)
    batch = reader.collect(cap)
    assert len(batch.content) <= 8000
    assert batch.collected_count == 3
    assert batch.remaining_count == 1


@pytest.mark.parametrize("runtime", [False, True])
def test_concurrent_last_slot_admission_is_atomic(runtime):
    store = MessageStore()
    for c in range(8 if runtime else 1):
        inbox = store.open_inbox(str(c))
        for child in range(4):
            sender = inbox.sender(identity(child))
            for _ in range(7 if c == (7 if runtime else 0) and child == 3 else 8):
                sender.send("a" * 2000)
    targets = [
        store.open_inbox(f"contender-{n}") if runtime else inbox for n in range(12)
    ]
    senders = [target.sender(identity(100 + n)) for n, target in enumerate(targets)]
    barrier = threading.Barrier(len(senders))

    def post(sender):
        barrier.wait(timeout=5)
        try:
            sender.send("winner")
            return "accepted"
        except MessageError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=len(senders)) as pool:
        outcomes = list(pool.map(post, senders))
    assert outcomes.count("accepted") == 1
    assert outcomes.count("queue_full") == 11
    assert sum(store.pending_counts().values()) == (256 if runtime else 32)
    store.close_inbox("0")
    assert sum(store.pending_counts().values()) == (224 if runtime else 0)
    senders[-1].close()


def test_discard_snapshot_racing_new_arrival_preserves_new_and_foreign():
    store, inbox, sender, reader = setup_queue()
    old = sender.send("old")
    snapshot_ids = [entry.message_id for entry in inbox.snapshot()]
    foreign = store.open_inbox("other")
    foreign_id = foreign.sender(identity()).send("foreign")
    barrier = threading.Barrier(2)

    def post():
        barrier.wait(timeout=5)
        return sender.send("new")

    def discard():
        barrier.wait(timeout=5)
        return inbox.discard(snapshot_ids + [foreign_id, old, "unknown"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        posted = pool.submit(post)
        discarded = pool.submit(discard)
        new_id = posted.result(timeout=5)
        assert discarded.result(timeout=5) == 1
    assert [m.message_id for m in inbox.snapshot()] == [new_id]
    assert foreign.snapshot()[0].message_id == foreign_id
    assert inbox.discard(snapshot_ids) == 0
    assert reader.collect().collected_count == 1
    assert store.pending_counts() == {"other": 1}


@pytest.mark.parametrize(
    "field,value",
    [
        ("handle_id", "h" * 129),
        ("run_id", "r" * 129),
        ("parent_run_id", "p" * 129),
        ("chain_id", "c" * 129),
        ("agent", "a" * 81),
        ("run_id", ""),
        ("agent", "bad\x1b"),
        ("chain_id", "bad\ud800"),
    ],
)
def test_unbounded_or_invalid_identity_cannot_create_sender(field, value):
    inbox = MessageStore().open_inbox("a")
    bad_identity = dataclasses.replace(identity(), **{field: value})
    assert_refusal("invalid_message", lambda: inbox.sender(bad_identity))
    inbox.sender(identity()).send("valid")
    assert len(inbox.snapshot()) == 1


def test_duplicate_sender_cannot_reset_lifetime_or_replace_live_identity():
    _, inbox, sender, _ = setup_queue()
    assert_refusal("unavailable", lambda: inbox.sender(identity()))
    assert_refusal(
        "unavailable",
        lambda: inbox.sender(dataclasses.replace(identity(), handle_id="different")),
    )
    sender.send("original still live")
    assert inbox.snapshot()[0].identity == identity()


def test_exact_serialized_envelope_boundary_counts_escaping(monkeypatch):
    from tldw_chatbook.Agents import fleet_messages

    monkeypatch.setattr(
        fleet_messages.uuid, "uuid4", lambda: type("ID", (), {"hex": "0" * 32})()
    )
    _, inbox, sender, reader = setup_queue()
    empty = {
        "message_id": "0" * 32,
        "handle_id": "h-0",
        "run_id": "r-0",
        "parent_run_id": "parent",
        "chain_id": "chain-a",
        "agent": "reader",
        "body": "",
    }
    overhead = len(
        json.dumps(empty, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    )
    quote_count = 4000 - overhead - 2000
    body = '"' * quote_count + "a" * (2000 - quote_count)
    assert len(body) == 2000
    sender.send(body)
    collected = json.loads(reader.collect().content)["messages"][0]
    assert (
        len(
            json.dumps(
                collected, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        )
        == 4000
    )
    assert collected["body"] == body
    assert_refusal("message_too_large", lambda: sender.send(body[:-1] + '"'))
    assert inbox.snapshot() == ()


def test_serialization_failure_does_not_remove_reports(monkeypatch):
    from tldw_chatbook.Agents import fleet_messages

    _, inbox, sender, reader = setup_queue()
    message_id = sender.send("preserve me")

    def fail_serialization(*args, **kwargs):
        raise ValueError("formatting failed")

    with monkeypatch.context() as patch:
        patch.setattr(fleet_messages.json, "dumps", fail_serialization)
        with pytest.raises(MessageError, match="invalid_message"):
            reader.collect()
    assert inbox.snapshot()[0].message_id == message_id
    assert reader.collect().collected_count == 1


def test_closed_sender_is_not_retained_by_admitted_reports():
    import gc
    import weakref

    _, inbox, sender, _ = setup_queue()
    sender.send("independent of lifetime accounting")
    ref = weakref.ref(sender)
    sender.close()
    del sender
    gc.collect()
    assert ref() is None
    assert inbox.snapshot()[0].body == "independent of lifetime accounting"
