"""Speech presentation never owns work or acknowledgement."""

import asyncio

import pytest

from tldw_chatbook.Persona_Buddy.speech import BuddySpeechItem, BuddySpeechQueue


def item(key, *, question=False, current=lambda: True):
    return BuddySpeechItem(key, "Research chat", key, question, current)


@pytest.mark.asyncio
async def test_named_serial_questions_first_and_duplicates_coalesce():
    spoken = []

    async def play(entry, current):
        assert current()
        spoken.append(entry.named_text)
        await asyncio.sleep(0)
        return True

    queue = BuddySpeechQueue(play)
    queue.pause()
    assert queue.enqueue(item("response"))
    assert not queue.enqueue(item("response"))
    queue.enqueue(item("question", question=True))
    queue.resume()
    await queue.wait_idle()
    assert spoken == ["Research chat. question", "Research chat. response"]
    await queue.aclose()


@pytest.mark.asyncio
async def test_pause_replays_current_skip_drops_only_current_mute_clears():
    entered = asyncio.Queue()
    release = asyncio.Event()
    stopped = []

    async def play(entry, current):
        await entered.put(entry.key)
        try:
            await release.wait()
            return current()
        finally:
            stopped.append(entry.key)

    queue = BuddySpeechQueue(play)
    queue.enqueue(item("one"))
    queue.enqueue(item("two"))
    assert await entered.get() == "one"
    queue.pause()
    await queue.wait_idle()
    assert stopped == ["one"]
    assert queue.state.paused and queue.state.queued == 2
    queue.resume()
    assert await entered.get() == "one"
    queue.skip()
    assert await entered.get() == "two"
    queue.mute()
    await queue.wait_idle()
    assert queue.state.muted and queue.state.queued == 0
    assert stopped == ["one", "one", "two"]
    await queue.aclose()


@pytest.mark.asyncio
async def test_stale_items_and_playback_failures_do_not_stall_queue():
    spoken = []

    async def play(entry, current):
        spoken.append(entry.key)
        if entry.key == "failure":
            raise ValueError("private provider detail")
        return True

    queue = BuddySpeechQueue(play)
    queue.pause()
    valid = [True]
    queue.enqueue(item("stale", current=lambda: valid[0]))
    valid[0] = False
    queue.enqueue(item("failure"))
    queue.enqueue(item("next"))
    queue.resume()
    await queue.wait_idle()
    assert spoken == ["failure", "next"]
    assert queue.state.error == "Speech could not finish."
    await queue.aclose()


@pytest.mark.asyncio
async def test_rebind_waits_for_old_playback_cleanup_before_new_item():
    events = []
    entered, cleanup = asyncio.Event(), asyncio.Event()

    async def play(entry, current):
        events.append(entry.key)
        if entry.key == "old":
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await cleanup.wait()
                events.append("old stopped")
        return True

    queue = BuddySpeechQueue(play)
    queue.enqueue(item("old"))
    await entered.wait()
    queue.reset()
    queue.enqueue(item("new"))
    await asyncio.sleep(0)
    assert events == ["old"]
    cleanup.set()
    await queue.wait_idle()
    assert events == ["old", "old stopped", "new"]
    await queue.aclose()


@pytest.mark.asyncio
async def test_microphone_hold_waits_for_output_cleanup_and_preserves_manual_pause():
    entered, cleanup = asyncio.Event(), asyncio.Event()
    spoken = []

    async def play(entry, current):
        spoken.append(entry.key)
        if len(spoken) == 1:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await cleanup.wait()
        return current()

    queue = BuddySpeechQueue(play)
    queue.enqueue(item("response"))
    await entered.wait()
    queue.set_input_active("mic", True)
    silent = asyncio.create_task(queue.wait_idle())
    await asyncio.sleep(0)
    assert not silent.done()
    cleanup.set()
    await silent
    assert queue.state.input_active and queue.state.queued == 1
    queue.pause()
    queue.set_input_active("mic", False)
    await asyncio.sleep(0)
    assert spoken == ["response"] and queue.state.paused
    queue.resume()
    await queue.wait_idle()
    assert spoken == ["response", "response"]
    await queue.aclose()


@pytest.mark.asyncio
async def test_skip_while_paused_removes_next_question_before_earlier_response():
    spoken = []

    async def play(entry, current):
        spoken.append(entry.key)
        return current()

    queue = BuddySpeechQueue(play)
    queue.pause()
    queue.enqueue(item("response"))
    queue.enqueue(item("question", question=True))
    queue.skip()
    queue.resume()
    await queue.wait_idle()
    assert spoken == ["response"]
    await queue.aclose()
