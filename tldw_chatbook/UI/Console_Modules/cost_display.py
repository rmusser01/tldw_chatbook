"""Shared original cost calculation; callers own checked inputs and native reads."""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

from ...Chat.console_chat_models import FEEDBACK_ACTIVE_RUN_STATUSES, ConsoleMessageRole, ConsoleRunStatus
from ...Chat.console_cost_tracker import ConsoleCacheState, fingerprint_break_reason, token_estimate_signature
from ...Chat.provider_readiness import provider_config_key
from ...Chat.provider_usage import ProviderUsage
from . import console_spend_projection as spend

CONSOLE_ACTIVE_RUN_STATUSES = FEEDBACK_ACTIVE_RUN_STATUSES


@dataclass(frozen=True)
class ConsoleCostFacts:
    snapshot: object
    cache_state: ConsoleCacheState
    break_reason: str | None
    projected_delta_usd: float | None
    ttl_remaining_s: float | None
    pricing_as_of: str | None
    pricing: object
    historical_media: bool
    tokenizer_ready: bool


def build_cost_facts(
    owner, *, session_id, store, controller, cost_controller, provider, model,
    tokenizer_ready, history_key, messages, history, catalog, helper_usage,
    fleet_tokens,
) -> ConsoleCostFacts | None:
    """Original settled/cache calculation, with original per-owner revision memos.

    Media and fingerprint readers may do native work. The selected backend calls
    this under its existing observed read, or proves those branches are cached.
    """
    projection_key = (
        history_key,
        provider,
        model,
        store.session_settings_revision(session_id),
        catalog,
        helper_usage,
        tokenizer_ready,
    )
    # PR2b Task 5 (cost rollup): the active conversation's LIVE
    # sub-agent fleet spend, folded into the snapshot's token total
    # (never priced -- see `ConsoleCostSnapshot.fleet_tokens`'s
    # docstring for why). Read straight off the SAME live source
    # the fleet rail rows themselves read
    # (`_console_agent_fleet_token_total` sums `bridge.fleet_
    # snapshot(...)`'s `FleetHandle.total_tokens`), so the chip and
    # the rail can never disagree about a conversation's fleet
    # spend.
    # PR3a-1 Task 6b (audit F3): plus whatever a SURVIVING child
    # billed after its turn's usage was attached to the assistant
    # message. PR3a-2 Task 3 (tasks 15660/15667): that spend is now
    # INTERIM, not lost -- when the conversation's last fleet child
    # settles, the controller's "usage-reattach" drain consumer
    # folds the whole turn (survivors included) back onto the
    # message's own usage row and this line falls to zero. Until
    # that drain, the money is named here on the chip's unpriced
    # sub-agent line. No double count: `unattributed_fleet_tokens`
    # counts ONLY payloads closed out after the latest attach (the
    # fold resets its watermark), and a live handle's
    # `FleetHandle.total_tokens` -- what `_console_agent_fleet_token_
    # total` sums -- is 0 until it finishes, by which point it has
    # left `fleet_snapshot`.
    if cost_controller is not None:
        unattributed = getattr(
            cost_controller, "unattributed_fleet_tokens", None
        )
        if callable(unattributed):
            fleet_tokens += int(unattributed(session_id) or 0)
    # task-15451: this method runs on the 0.2s tick for the whole
    # duration of a run (plus every control-bar sync pass and the
    # 10s TTL timer), and the equality guard in
    # `_sync_console_cost_chip` gates only the REPAINT -- the build
    # itself always ran. Without the memo below every usage-less row
    # (all user/system rows, legacy assistant rows, the staged
    # evidence pseudo-row) was re-tokenized by a per-character
    # Python loop 5x/s: ~28ms/tick on a 99KB transcript, measured.
    # Row estimates are verified against detached text when a
    # display revision changes. Unchanged ticks reuse the settled
    # aggregate without traversing every row.
    #
    # Gating the whole snapshot on `store.payload_revision` instead
    # was considered and rejected: usage is not payload-affecting, so
    # `ConsoleChatStore.set_message_usage` never bumps that counter --
    # a real priced usage landing on an ALREADY-terminal row (the
    # documented Stop-path ordering) would leave the chip showing the
    # estimated total until some unrelated edit moved the revision.
    # ADR-190's display revision observes that late usage separately.
    estimate_cache = owner._console_cost_estimate_cache_or_new()
    cached_spend = getattr(owner, "_console_settled_spend_cache", None)
    if cached_spend is None or cached_spend[0] != projection_key:
        snapshot_messages = spend.build_console_current_cost_messages(
            messages, history.current_ids
        )
        if not tokenizer_ready and any(
            not isinstance(message.usage, ProviderUsage)
            and str(message.content or "").strip()
            for message in snapshot_messages
        ):
            return None  # A partial measured total must not stand in for Current.
        settled_snapshot = owner._build_cost_snapshot(
            snapshot_messages,
            provider=provider,
            model=model,
            fleet_tokens=0,
            estimate_cache=estimate_cache,
            auxiliary_usage=tuple(record.usage for record in helper_usage),
            **spend.pricing_snapshot_options(catalog),
        )
        historical_media = (
            tokenizer_ready
            and any(
                message.role is ConsoleMessageRole.USER
                and message.attachments
                and message.id in history.request_ids
                for message in messages
            )
            and any(
                row.attachments
                for row in controller._lightweight_provider_message_rows(
                    [
                        message
                        for message in messages
                        if message.id in history.request_ids
                    ],
                    skip_failed=True,
                    session_id=session_id,
                    turn_context=controller.resolve_turn_configuration_snapshot(
                        session_id
                    ),
                )
            )
        )
        owner._console_settled_spend_cache = (
            projection_key,
            snapshot_messages,
            settled_snapshot,
            historical_media,
        )
    else:
        snapshot_messages = cached_spend[1]
        settled_snapshot = cached_spend[2]
        historical_media = cached_spend[3]
    fleet_tokens = max(0, fleet_tokens)
    snapshot = replace(
        settled_snapshot,
        total_tokens=settled_snapshot.total_tokens + fleet_tokens,
        fleet_tokens=fleet_tokens,
    )

    controller = cost_controller
    run_status = (
        controller.run_state_for(session_id).status
        if controller is not None
        else ConsoleRunStatus.IDLE
    )
    # Fingerprint compare is the expensive step (rebuilds the
    # pre-compaction provider payload) -- only pay it when the
    # session isn't actively streaming AND its payload has actually
    # changed since the last check (revision `!=`, not `>`: a
    # restore can reset the store's counter back down).
    break_reason = owner._console_cost_break_reasons.get(session_id)
    if (
        tokenizer_ready
        and controller is not None
        and run_status not in CONSOLE_ACTIVE_RUN_STATUSES
    ):
        current_revision = store.payload_revision(session_id)
        if current_revision != owner._console_cost_fp_revisions.get(session_id):
            baseline = controller.payload_fingerprint_baseline(session_id)
            break_reason = None
            if baseline is not None:
                current_fp = controller.compute_current_fingerprint(session_id)
                break_reason = fingerprint_break_reason(baseline, current_fp)
            owner._console_cost_fp_revisions[session_id] = current_revision
            owner._console_cost_break_reasons[session_id] = break_reason

    cache_state = ConsoleCacheState.NONE
    ttl_remaining_s: float | None = None
    if controller is not None:
        warm_until, had_activity = controller.cache_ttl_snapshot(session_id)
        if had_activity and warm_until is not None:
            now = time.monotonic()
            if now < warm_until:
                cache_state = ConsoleCacheState.WARM
                ttl_remaining_s = warm_until - now
            else:
                cache_state = ConsoleCacheState.EXPIRED
    owner._console_cost_cache_state = cache_state

    projected_delta_usd: float | None = None
    pricing_as_of: str | None = None
    provider_key = provider_config_key(provider)
    pricing = catalog.get_pricing(provider_key, model or "")
    if pricing is not None:
        pricing_as_of = pricing.as_of
        if (
            tokenizer_ready
            and cache_state == ConsoleCacheState.WARM
            and break_reason
            and pricing.cache_write_per_mtok is not None
            and pricing.cache_read_per_mtok is not None
        ):
            # `break_reason` is required here (Qodo round, finding
            # 3): `build_cost_state`/`_cache_state_line` only ever
            # read `projected_delta_usd` inside their own
            # `break_reason`-gated branches (the alert suffix on the
            # label, and the "~+$" clause in the tooltip's cache
            # line) -- with no break reason the value is built and
            # then silently discarded every call. Skipping the
            # (expensive: `_estimate_tokens_locally` over the WHOLE
            # transcript) computation here avoids that on every
            # 0.2s/10s sync tick for a long-running WARM session
            # that never alerts.
            #
            # `snapshot_messages` (not `messages`): same mid-stream-
            # animation guard as the snapshot above -- an in-flight
            # row's growing content must not grow the projected
            # break-delta either, e.g. when a NEW turn starts
            # streaming while a PRIOR turn's alert is still showing
            # (fingerprint recompute -- and so `break_reason` /
            # `alert` -- is frozen during the run, but this
            # projection is computed fresh every call).
            #
            # ADR-190: unchanged display revisions reuse this total
            # directly. A changed transcript verifies the row
            # signature through the existing token-estimate memo.
            projection_tokens_key = (projection_key, provider_key)
            cached_tokens = getattr(
                owner, "_console_cost_projection_tokens_cache", None
            )
            if (
                cached_tokens is not None
                and cached_tokens[0] == projection_tokens_key
            ):
                estimated_tokens = cached_tokens[1]
            else:
                projection_rows = tuple(
                    (
                        str(getattr(message.role, "value", message.role)),
                        message.content,
                    )
                    for message in snapshot_messages
                )

                def _estimate_projection() -> int:
                    return owner._estimate_tokens_locally(
                        [
                            {"role": role, "content": content}
                            for role, content in projection_rows
                        ],
                        model or "",
                        provider_key,
                    )

                projection_cache = owner._console_cost_estimate_cache_or_new()
                estimated_tokens = projection_cache.estimate(
                    ("#cost-projection", session_id),
                    token_estimate_signature(
                        projection_rows, model or "", provider_key
                    ),
                    _estimate_projection,
                )
                owner._console_cost_projection_tokens_cache = (
                    projection_tokens_key,
                    estimated_tokens,
                )
            rate_delta = (
                pricing.cache_write_per_mtok - pricing.cache_read_per_mtok
            ) / 1_000_000
            projected_delta_usd = round(estimated_tokens * rate_delta, 6)

    return ConsoleCostFacts(
        snapshot, cache_state, break_reason, projected_delta_usd, ttl_remaining_s,
        pricing_as_of, pricing, historical_media, tokenizer_ready,
    )


def build_cost_display(facts, *, context_state, pending_attachments, draft_text, rate_limit_line):
    """Use the original metric/label builder; no alternative cost arithmetic."""
    if facts is None:
        return None
    return spend.build_console_spend_cost_state(
        facts.snapshot, facts.cache_state, facts.break_reason,
        facts.projected_delta_usd, facts.ttl_remaining_s, facts.pricing_as_of,
        facts.pricing is not None,
        context_state if facts.tokenizer_ready else None,
        facts.historical_media, pending_attachments,
        facts.pricing.input_per_mtok if facts.pricing is not None else None,
        draft_text, rate_limit_line,
    )
