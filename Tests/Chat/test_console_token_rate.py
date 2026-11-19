"""Unit tests for the Console tokens-per-second tracker (TASK-35100).

Covers ``Chat/console_token_rate.py``: the live EMA rate during a streamed
response, the provider-accurate overall figure at settle (measured from the
first content observation, so time-to-first-token is excluded), and the
edge cases the display depends on -- missing usage, late usage, message
switches, stalls, and non-monotonic clocks. The chip rendering contract
(``format_rate_chip_live_fragment`` / ``format_rate_chip_settled``) are
pinned here too, so the widget keeps no formatting logic of its own.

The token counter is injected, so no test depends on tokenizer behavior.
The tracker is fed on every poll tick (streaming or not), so a stalled
endpoint's rate decays toward zero through the same math as any other
tick -- there is no separate stall path.
"""

import math

import pytest

from tldw_chatbook.Chat.console_token_rate import (
    STALE_AFTER_SECONDS,
    TokenRateReading,
    TokenRateTracker,
    format_rate_chip_live_fragment,
    format_rate_chip_settled,
    merge_rate_into_run_copy,
)

APPROX = pytest.approx


def _count(text: str) -> int:
    """Deterministic stand-in tokenizer: one token per four characters."""
    return len(text) // 4


def _tracker(**kwargs) -> TokenRateTracker:
    kwargs.setdefault("count_fn", _count)
    return TokenRateTracker(**kwargs)


def _alpha(tracker: TokenRateTracker, dt: float) -> float:
    """The update weight the implementation uses for one dt step."""
    return 1.0 - 0.5 ** (dt / tracker.half_life)


class TestLiveRate:
    def test_first_content_observation_reports_hidden(self) -> None:
        tracker = _tracker()
        reading = tracker.observe("m1", status="streaming", content="xxxx", now=100.0)
        assert reading.kind == "hidden"

    def test_pre_content_streaming_reports_hidden_and_skips_clock(self) -> None:
        tracker = _tracker()
        assert (
            tracker.observe("m1", status="streaming", content="", now=100.0).kind
            == "hidden"
        )
        # The empty window must not count as token-generation time: the
        # rate measured at the next tick starts from THIS moment, not 3s
        # earlier while the endpoint was still processing the prompt.
        assert (
            tracker.observe("m1", status="streaming", content="x" * 400, now=103.0).kind
            == "hidden"
        )
        reading = tracker.observe(
            "m1", status="streaming", content="x" * 800, now=104.0
        )
        assert reading.kind == "live"
        assert reading.rate == APPROX(100.0)

    def test_steady_stream_converges_to_true_rate(self) -> None:
        tracker = _tracker()
        # 100 tok/s: 100 tokens every 1.0s. First observation seeds the
        # baseline, every later tick confirms the rate.
        for now, tokens in [(0.0, 0), (1.0, 100), (2.0, 200), (3.0, 300)]:
            reading = tracker.observe(
                "m1", status="streaming", content="x" * (tokens * 4), now=now
            )
        assert reading.kind == "live"
        assert reading.rate == APPROX(100.0)

    def test_ema_smooths_a_burst_instead_of_snapping(self) -> None:
        tracker = _tracker(half_life=1.0)
        for now, tokens in [(0.0, 0), (1.0, 100), (2.0, 200)]:
            tracker.observe(
                "m1", status="streaming", content="x" * (tokens * 4), now=now
            )
        # Steady 100 tok/s, then a burst: 400 more tokens in one 1.0s tick
        # (500 tok/s raw). With half_life=1.0 the tick carries exactly half
        # the new weight.
        reading = tracker.observe("m1", status="streaming", content="x" * 2400, now=3.0)
        assert reading.kind == "live"
        # ema was 100 (set raw on the first live tick at t=2); the burst
        # tick blends half of its raw 500: 100 + 0.5 * (500 - 100) = 250.
        assert reading.rate == APPROX(250.0)

    def test_default_half_life_is_one_second(self) -> None:
        assert TokenRateTracker(count_fn=_count).half_life == 1.0

    def test_alpha_uses_half_life_semantics(self) -> None:
        tracker = _tracker(half_life=2.0)
        # After dt == half_life the old estimate carries exactly half.
        assert _alpha(tracker, 2.0) == APPROX(0.5)


class TestSettle:
    def test_settle_uses_provider_tokens_from_first_content_time(self) -> None:
        tracker = _tracker()
        # Content first observed at t=1.0 (the 0.0->1.0 window was prompt
        # processing); stream ends at t=11.0 with 800 provider tokens.
        for now, tokens in [(0.0, 0), (1.0, 100), (2.0, 200)]:
            tracker.observe(
                "m1", status="streaming", content="x" * (tokens * 4), now=now
            )
        reading = tracker.observe(
            "m1",
            status="complete",
            content="x" * 3000,
            now=11.0,
            output_tokens=800,
        )
        assert reading.kind == "settled"
        assert reading.total_tokens == 800
        assert reading.elapsed_seconds == APPROX(10.0)
        assert reading.rate == APPROX(80.0)

    def test_settle_without_usage_falls_back_to_estimate(self) -> None:
        tracker = _tracker()
        for now, tokens in [(0.0, 0), (1.0, 100), (2.0, 200)]:
            tracker.observe(
                "m1", status="streaming", content="x" * (tokens * 4), now=now
            )
        reading = tracker.observe("m1", status="complete", content="x" * 1200, now=4.0)
        assert reading.kind == "settled"
        assert reading.total_tokens == 300
        assert reading.elapsed_seconds == APPROX(3.0)
        assert reading.rate == APPROX(100.0)

    def test_terminal_without_streaming_baseline_hides(self) -> None:
        # A response that completed entirely between two poll ticks (or a
        # transcript seen for the first time after it finished): the tracker
        # never watched it stream, so it must NOT manufacture an elapsed
        # window measured from its own first sight.
        tracker = _tracker()
        reading = tracker.observe(
            "m1",
            status="complete",
            content="x" * 4000,
            now=9.0,
            output_tokens=1000,
        )
        assert reading.kind == "hidden"

    def test_late_provider_usage_upgrades_settled_total(self) -> None:
        tracker = _tracker()
        for now, tokens in [(0.0, 0), (1.0, 100), (2.0, 200)]:
            tracker.observe(
                "m1", status="streaming", content="x" * (tokens * 4), now=now
            )
        settled = tracker.observe("m1", status="complete", content="x" * 1200, now=4.0)
        assert settled.total_tokens == 300
        # Usage attaches after the terminal paint (store display-revision
        # bump): the settled total upgrades, the elapsed window does not move.
        upgraded = tracker.observe(
            "m1",
            status="complete",
            content="x" * 1200,
            now=99.0,
            output_tokens=400,
        )
        assert upgraded.kind == "settled"
        assert upgraded.total_tokens == 400
        assert upgraded.elapsed_seconds == APPROX(3.0)
        assert upgraded.rate == APPROX(400.0 / 3.0)

    def test_zero_elapsed_settle_reports_tokens_without_rate(self) -> None:
        tracker = _tracker()
        tracker.observe("m1", status="streaming", content="x" * 400, now=5.0)
        reading = tracker.observe(
            "m1",
            status="complete",
            content="x" * 400,
            now=5.0,
            output_tokens=100,
        )
        assert reading.kind == "settled"
        assert reading.total_tokens == 100
        assert reading.rate is None


class TestLifecycle:
    def test_new_message_id_resets_and_hides(self) -> None:
        tracker = _tracker()
        for now, tokens in [(0.0, 0), (1.0, 100)]:
            tracker.observe(
                "m1", status="streaming", content="x" * (tokens * 4), now=now
            )
        tracker.observe("m1", status="complete", content="x" * 400, now=2.0)
        reading = tracker.observe("m2", status="streaming", content="x" * 400, now=50.0)
        assert reading.kind == "hidden"
        # The old turn's settle must be gone: a later terminal observation
        # of m1 cannot resurrect it.
        stray = tracker.observe("m1", status="complete", content="", now=51.0)
        assert stray.kind == "hidden"

    def test_next_stream_replaces_settled_reading(self) -> None:
        tracker = _tracker()
        for now, tokens in [(0.0, 0), (1.0, 100)]:
            tracker.observe(
                "m1", status="streaming", content="x" * (tokens * 4), now=now
            )
        tracker.observe("m1", status="complete", content="x" * 400, now=2.0)
        # A new turn on a new message id: fresh window, no memory of m1.
        assert (
            tracker.observe("m2", status="streaming", content="", now=50.0).kind
            == "hidden"
        )
        assert (
            tracker.observe("m2", status="streaming", content="x" * 800, now=51.0).kind
            == "hidden"
        )
        reading = tracker.observe(
            "m2", status="streaming", content="x" * 1200, now=52.0
        )
        assert reading.kind == "live"
        # Baseline was 200 tokens at t=51; 300 at t=52 -> 100 tok/s.
        assert reading.rate == APPROX(100.0)

    def test_reset_clears_everything(self) -> None:
        tracker = _tracker()
        tracker.observe("m1", status="streaming", content="x" * 400, now=0.0)
        tracker.observe("m1", status="streaming", content="x" * 800, now=1.0)
        tracker.reset()
        reading = tracker.observe("m1", status="streaming", content="x" * 800, now=9.0)
        assert reading.kind == "hidden"

    def test_no_message_at_all_hides(self) -> None:
        tracker = _tracker()
        reading = tracker.observe(None, status="pending", content="", now=1.0)
        assert reading.kind == "hidden"


class TestClockEdges:
    def test_non_monotonic_clock_keeps_previous_rate(self) -> None:
        tracker = _tracker()
        for now, tokens in [(10.0, 0), (11.0, 100), (12.0, 200)]:
            tracker.observe(
                "m1", status="streaming", content="x" * (tokens * 4), now=now
            )
        before = tracker.observe("m1", status="streaming", content="x" * 800, now=13.0)
        assert before.kind == "live"
        # A tick stamped earlier than the last one (timer overlap): hold.
        after = tracker.observe("m1", status="streaming", content="x" * 1200, now=12.5)
        assert after.kind == "live"
        assert after.rate == before.rate

    def test_stall_decays_rate_toward_zero_and_recovers(self) -> None:
        tracker = _tracker()
        for now, tokens in [(0.0, 0), (1.0, 100), (2.0, 200)]:
            tracker.observe(
                "m1", status="streaming", content="x" * (tokens * 4), now=now
            )
        # The endpoint stalls: ticks keep arriving, tokens stop growing.
        reading = tracker.observe("m1", status="streaming", content="x" * 800, now=32.0)
        assert reading.kind == "live"
        assert reading.rate is not None
        assert reading.rate < 5.0
        # Generation resumes at 100 tok/s: within one window it is visibly
        # recovering, not stuck at zero and not snapping to a lie.
        reading = tracker.observe(
            "m1", status="streaming", content="x" * 1200, now=33.0
        )
        assert reading.kind == "live"
        assert reading.rate == APPROX(100.0 * _alpha(tracker, 1.0))

    def test_ui_freeze_gap_stays_finite_and_recovers(self) -> None:
        tracker = _tracker()
        for now, tokens in [(0.0, 0), (1.0, 100)]:
            tracker.observe(
                "m1", status="streaming", content="x" * (tokens * 4), now=now
            )
        # The poll stopped for 30s while the endpoint kept generating: the
        # only honest wall-window figure is tokens-arrived over the whole
        # gap (750 tok / 30 s = 25), finite and free of divide-by-zero.
        reading = tracker.observe(
            "m1", status="streaming", content="x" * 3400, now=31.0
        )
        assert reading.kind == "live"
        assert reading.rate == APPROX(25.0)
        assert math.isfinite(reading.rate)

    def test_tracker_marks_reading_stale_after_growth_gap(self) -> None:
        """Review round 2: growth stops, and STALE_AFTER_SECONDS later the
        reading flags itself stale (the fragment suppressor)."""
        tracker = _tracker()
        for now, tokens in [(0.0, 0), (1.0, 100), (2.0, 200)]:
            tracker.observe(
                "m1", status="streaming", content="x" * (tokens * 4), now=now
            )
        assert tracker._last_growth_at == 2.0
        reading = tracker.observe(
            "m1",
            status="streaming",
            content="x" * 800,
            now=2.0 + STALE_AFTER_SECONDS + 0.1,
        )
        assert reading.stale is True
        # One gap tick decays the EMA partway (100 -> ~23 at half-life 1);
        # the exact value is the EMA's business -- the flag is the contract.
        assert reading.rate is not None and 0.0 <= reading.rate < 100.0

    def test_token_count_regression_never_reports_negative_rate(self) -> None:
        tracker = _tracker()
        for now, tokens in [(0.0, 0), (1.0, 100)]:
            tracker.observe(
                "m1", status="streaming", content="x" * (tokens * 4), now=now
            )
        # Content shrank (an unusual store rewrite): clamp, don't lie.
        reading = tracker.observe("m1", status="streaming", content="x" * 40, now=2.0)
        assert reading.kind == "live"
        assert reading.rate >= 0.0


class TestLiveFragment:
    """The run-chip fragment (redesign 2026-10-16): the live rate rides the
    EXISTING run chip copy, so the strip never gains a slot while streaming."""

    def test_live_fragment_formats_one_decimal(self) -> None:
        assert (
            format_rate_chip_live_fragment(TokenRateReading(kind="live", rate=42.34))
            == "42.3 tok/s"
        )

    def test_stale_fragment_is_suppressed(self) -> None:
        """Review round 2: tool rounds and approval waits stop token growth
        without ending the run; a stale fragment would mislabel the pause
        as crawling streaming. A genuinely slow endpoint that keeps
        trickling stays fresh and visible (no rate floor)."""
        assert (
            format_rate_chip_live_fragment(
                TokenRateReading(kind="live", rate=40.0, stale=True)
            )
            == ""
        )
        assert (
            format_rate_chip_live_fragment(
                TokenRateReading(kind="live", rate=0.3, stale=False)
            )
            == "0.3 tok/s"
        )

    def test_non_live_readings_have_no_fragment(self) -> None:
        assert format_rate_chip_live_fragment(TokenRateReading(kind="hidden")) == ""
        assert (
            format_rate_chip_live_fragment(
                TokenRateReading(kind="settled", rate=80.0)
            )
            == ""
        )
        assert format_rate_chip_live_fragment(TokenRateReading(kind="live")) == ""


class TestRunCopyMerge:
    """The pure merge decision (review round 2): the live fragment joins
    the run copy ONLY while the run streams, no approval is parked, and
    the reading is fresh."""

    FRESH = TokenRateReading(kind="live", rate=42.0)

    def test_happy_path_merges(self) -> None:
        assert (
            merge_rate_into_run_copy(
                "Streaming response.", self.FRESH,
                run_streaming=True, approval_pending=False,
            )
            == "Streaming response. · 42.0 tok/s"
        )

    def test_approval_pending_leaves_copy_untouched(self) -> None:
        assert (
            merge_rate_into_run_copy(
                "Waiting for approval on round 2.", self.FRESH,
                run_streaming=True, approval_pending=True,
            )
            == "Waiting for approval on round 2."
        )

    def test_not_streaming_leaves_copy_untouched(self) -> None:
        assert (
            merge_rate_into_run_copy(
                "Agent running.", self.FRESH,
                run_streaming=False, approval_pending=False,
            )
            == "Agent running."
        )

    def test_stale_reading_leaves_copy_untouched(self) -> None:
        assert (
            merge_rate_into_run_copy(
                "Streaming response.",
                TokenRateReading(kind="live", rate=40.0, stale=True),
                run_streaming=True, approval_pending=False,
            )
            == "Streaming response."
        )

    def test_no_copy_never_grows_one(self) -> None:
        assert (
            merge_rate_into_run_copy(
                "", self.FRESH, run_streaming=True, approval_pending=False
            )
            == ""
        )


class TestSettledBadge:
    """The settled badge replaces the run chip's slot after it hides: compact
    label, detail (tokens, elapsed) demoted to the tooltip."""

    def test_settled_label_is_compact_and_marked_avg(self) -> None:
        visible, label, tooltip = format_rate_chip_settled(
            TokenRateReading(
                kind="settled", rate=80.0, total_tokens=800, elapsed_seconds=10.0
            )
        )
        assert visible is True
        assert label == "80.0 tok/s avg"
        assert "800 tok" in tooltip
        assert "10.0s" in tooltip

    def test_settled_without_rate_still_shows_tokens(self) -> None:
        visible, label, tooltip = format_rate_chip_settled(
            TokenRateReading(kind="settled", rate=None, total_tokens=9500)
        )
        assert visible is True
        assert label == "9.5k tok"
        assert "tok/s" not in label

    def test_settled_without_anything_is_not_visible(self) -> None:
        assert format_rate_chip_settled(TokenRateReading(kind="settled")) == (
            False,
            "",
            "",
        )

    def test_non_settled_readings_are_not_visible(self) -> None:
        for kind in ("hidden", "live"):
            assert format_rate_chip_settled(TokenRateReading(kind=kind)) == (
                False,
                "",
                "",
            )
