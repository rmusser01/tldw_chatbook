"""Measure actual old/new output using separate, verified stereo channels."""

import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path
import time

import controlled_instrument as ci
from windows_controlled_adapter import Capture, ControlledWavPlayback, q
from controlled_instrument import channel_packets, separation_check
from tldw_chatbook.TTS.audio_output_ownership import AudioOutputOwnership
from tldw_chatbook.TTS.audio_output_runs import PcmOutputRun
from tldw_chatbook.TTS.controlled_wav_profile import WavBookmark
from tldw_chatbook.Audio.streaming_sink import pump


async def pcm_feed(run, pcm):
    async def chunks():
        for at in range(0, len(pcm), 960 * 4):
            yield pcm[at : at + 960 * 4]
            await asyncio.sleep(0)

    return await pump(run.sink, chunks())


async def prepared(kind, artifact, events):
    if kind == "controlled":
        return await ControlledWavPlayback.prepare(
            artifact, WavBookmark(artifact.sha256, q.RATE, 0)
        )
    return PcmOutputRun(
        q.RATE,
        2,
        on_event=lambda event: events.append(
            (type(event).__name__, time.perf_counter())
        ),
    )


async def start_run(run, kind, lease, pcm):
    started = await run.start(lease)
    if kind == "pcm":
        if not started:
            raise RuntimeError("pcm_start_refused")
        return asyncio.create_task(pcm_feed(run, pcm))
    return None


async def result_of(run, kind, feed):
    if kind == "controlled":
        return asdict(await asyncio.wait_for(run.completion, 12))
    result = await asyncio.wait_for(asyncio.shield(feed), 12)
    return {"pump": asdict(result), "native_retirement": run.retirement_confirmed}


async def transfer(directory, old_kind, new_kind, phase, ordinal):
    old_artifact, old_source, old_pcm = ci.make_fixture(
        directory, 2.5, 41177 + ordinal, 0
    )
    new_artifact, new_source, new_pcm = ci.make_fixture(
        directory, 0.5, 51177 + ordinal, 1
    )
    events_old, events_new = [], []
    old = new = None
    capture = Capture(7)
    host = AudioOutputOwnership()
    row = {
        "old_kind": old_kind,
        "new_kind": new_kind,
        "phase": phase,
        "ordinal": ordinal,
    }
    row["old_fixture"] = ci.fixture_receipt(old_artifact)
    row["new_fixture"] = ci.fixture_receipt(new_artifact)
    old_feed = new_feed = None
    try:
        old = await prepared(old_kind, old_artifact, events_old)
        new = await prepared(new_kind, new_artifact, events_new)
        await capture.start()
        await asyncio.sleep(0.45)
        baseline_end = time.perf_counter() - 0.05
        row["baseline"] = capture.quiet(capture.ready_at, baseline_end)
        if row["baseline"]["status"] != "pass":
            return row
        old_lease = await host.acquire("old source", old.retire)
        old_started = time.perf_counter()
        old_feed = await start_run(old, old_kind, old_lease, old_pcm)
        deadline = time.perf_counter() + 8
        while True:
            if phase == "startup":
                if time.perf_counter() - old_started >= 0.020:
                    break
            elif old_kind == "controlled":
                observation = old.position()
                if observation and observation.frame >= (
                    old_artifact.content_frames
                    if phase == "padding"
                    else round(0.5 * q.RATE)
                ):
                    break
            elif time.perf_counter() - old_started >= 0.65:
                break
            if time.perf_counter() >= deadline:
                raise RuntimeError("transfer_trigger_timeout")
            await asyncio.sleep(0.002)
        row["acquire_requested_at_s"] = time.perf_counter()
        new_lease = await host.acquire("successor source", new.retire)
        row["successor_granted_at_s"] = time.perf_counter()
        row["old_lease_revoked"] = old_lease.revoked
        new_feed = await start_run(new, new_kind, new_lease, new_pcm)
        row["old_completion"] = await result_of(old, old_kind, old_feed)
        row["new_completion"] = await result_of(new, new_kind, new_feed)
        row["new_retirement_confirmed"] = await new.retire()
        await asyncio.sleep(0.45)
    except Exception as error:
        row["error"] = type(error).__name__
    finally:
        row["old_retirement_confirmed"] = old is not None and await old.retire()
        row["new_retirement_confirmed"] = new is not None and await new.retire()
        row["old_fixture_mode_after_retirement"] = ci.fixture_mode(old_artifact.path)
        row["new_fixture_mode_after_retirement"] = ci.fixture_mode(new_artifact.path)
        for feed in (old_feed, new_feed):
            if feed is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(feed), 2)
                except (Exception, asyncio.CancelledError):
                    row["feed_unsettled"] = True
        row["capture_reaped"] = await capture.finish()
        row["owner_released"] = host.owner is None
        row["capture_configuration"] = capture.config
        row["capture_metrics"] = capture.metrics
        row["capture_error"] = capture.error
    row["old_events"], row["new_events"] = events_old, events_new
    if row.get("error") or row.get("capture_error"):
        return row
    old_match, old_diagnostics = q._match_packets(
        channel_packets(capture.packets, 0), old_source
    )
    new_match, new_diagnostics = q._match_packets(
        channel_packets(capture.packets, 1), new_source
    )
    row.update(
        old_match=old_match,
        new_match=new_match,
        old_diagnostics=old_diagnostics,
        new_diagnostics=new_diagnostics,
    )
    if old_match and new_match:
        row["old_end_to_successor_onset_s"] = (
            new_match["capture_onset_qpc_s"] - old_match["capture_end_qpc_s"]
        )
        row["new_source_start_error_s"] = new_match["start_source_s"]
        row["new_source_end_error_s"] = new_match["end_source_s"] - 0.5
        row["right_channel_during_old_only"] = q.quiet_window(
            channel_packets(capture.packets, 1),
            capture.polls,
            old_match["capture_onset_qpc_s"] + 0.05,
            min(old_match["capture_end_qpc_s"], new_match["capture_onset_qpc_s"])
            - 0.05,
            max_gap_s=min(0.15, capture.config["actual_buffer_frames"] / q.RATE),
            drain_delay_s=capture.config["device_period_s"],
        )
        row["left_channel_during_new_only"] = q.quiet_window(
            channel_packets(capture.packets, 0),
            capture.polls,
            max(old_match["capture_end_qpc_s"], new_match["capture_onset_qpc_s"])
            + 0.05,
            new_match["capture_end_qpc_s"] - 0.05,
            max_gap_s=min(0.15, capture.config["actual_buffer_frames"] / q.RATE),
            drain_delay_s=capture.config["device_period_s"],
        )
    elif new_match:
        row["old_channel_monitor"] = q.quiet_window(
            channel_packets(capture.packets, 0),
            capture.polls,
            capture.ready_at,
            new_match["capture_end_qpc_s"] + 0.1,
            max_gap_s=min(0.15, capture.config["actual_buffer_frames"] / q.RATE),
            drain_delay_s=capture.config["device_period_s"],
        )
    return row


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--play", required=True, action="store_true")
    parser.add_argument("--case-index", type=int, choices=range(6))
    parser.add_argument("--rounds", type=int, default=3, choices=range(1, 4))
    args = parser.parse_args()
    output, metadata = ci.new_output(args.output_dir)
    result = {
        **metadata,
        "separation_self_check_gaps_s": separation_check(),
        "runs": [],
    }
    cases = [
        ("controlled", "controlled", "active"),
        ("controlled", "controlled", "startup"),
        ("controlled", "controlled", "padding"),
        ("controlled", "pcm", "active"),
        ("pcm", "controlled", "active"),
        ("pcm", "pcm", "active"),
    ]
    if args.case_index is not None:
        cases = [cases[args.case_index]]
    fixtures = ci.Fixtures()
    clean = False
    try:
        for round_number in range(1, args.rounds + 1):
            for index, case in enumerate(cases, 1):
                clean = False
                ordinal = (round_number - 1) * len(cases) + index
                row = await transfer(fixtures.directory, *case, ordinal)
                row["round"] = round_number
                result["runs"].append(row)
                ci.save(output / "transfers.json", result)
                clean = all(
                    row.get(key) is True
                    for key in (
                        "old_retirement_confirmed",
                        "new_retirement_confirmed",
                        "capture_reaped",
                        "owner_released",
                    )
                ) and not row.get("feed_unsettled")
                if (
                    not clean
                    or row.get("error")
                    or row.get("capture_error")
                    or row.get("baseline", {}).get("status") != "pass"
                ):
                    return 1
    finally:
        if clean:
            fixtures.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
