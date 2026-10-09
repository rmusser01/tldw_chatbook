"""The Console store's app-exit teardown (``ConsoleChatStore.end_app_runtime``).

Moved out of ``console_chat_store.py``, which is held by a size ratchet
(``Tests/Architecture/test_module_size_ratchet.py``), by TASK-33628.5. That
task added the one variation: ``replace_state=False``.

The teardown closes the Canvas runtimes, permanently fences, settles and
drains the provider-trace settlements, closes their executor, shuts the
stream-persistence executor down and retries failed settlements once more.
It also drops the volatile dispatch-recovery projections, a wholesale state
replacement, so normally the whole teardown runs inside the store's
voice-promotion replacement fence. That fence refuses while any store
mutation holds its promotion admission.

A Console Delete or Undo saving off the event loop holds that admission
until its result is applied. App teardown waits a bounded time for such a
save (``console_durable_writes``); if the save outlasts the bound,
``replace_state=False`` keeps the projections and skips the fence, so the
rest of the teardown still runs instead of being refused with it. Nothing
else in it depends on the fence.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any


def end_store_runtime(store: Any, *, replace_state: bool) -> None:
    """Run the store's app-exit teardown.

    Args:
        store: The ``ConsoleChatStore`` being ended.
        replace_state: Also drop the volatile recovery projections, under
            the voice-promotion replacement fence. False skips only that.

    Raises:
        RuntimeError: ``replace_state`` and the fence refused (voice
            promotion state still holds the store).
    """
    fence = (
        store._voice_promotion_state_replacement_scope()
        if replace_state
        else nullcontext()
    )
    with fence:
        # A retained ordinary Send commit must physically settle before close.
        with store._preparation_lock:
            if (
                store._native_commit_owners_by_preparation
                or store._durable_commit_in_flight
            ):
                raise RuntimeError("Durable acceptance commit is still owned.")
        participant = store.canvas_promotion_participant
        if participant is not None:
            participant.close_runtime()
        turns = store.canvas_turn_controller
        if turns is not None and turns is not participant:
            turns.close_runtime()

        with store._fence_provider_trace_settlement_registrations((), permanent=True):
            store._settle_all_provider_trace_settlements()
            store._drain_retained_provider_trace_settlements_on_teardown()
            if replace_state:
                with store._preparation_lock:
                    store._dispatch_recoveries_by_session.clear()
                    store._dispatch_recovery_message_baselines.clear()
                    store._dispatch_recovery_generation_tokens.clear()
                    store._dispatch_recovery_queue_hydration_pending.clear()
        store._close_provider_trace_settlement_executor()
        with store._stream_persistence_deferred_lock:
            store._stream_persistence_executor_closed = True
        store._stream_persistence_executor.shutdown(wait=True)
        store._retry_failed_provider_trace_settlements_on_teardown()
