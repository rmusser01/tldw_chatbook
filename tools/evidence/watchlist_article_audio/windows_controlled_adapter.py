"""Native adapter controls; numerical receipts only, no retained capture PCM."""

import argparse
import asyncio
from dataclasses import asdict
import json
import os
import multiprocessing as mp
from pathlib import Path
import stat
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/evidence/watchlist_article_audio")]
import qualification as q
import controlled_instrument as ci
from tldw_chatbook.TTS.audio_output_ownership import AudioOutputOwnership
from tldw_chatbook.TTS.controlled_wav_playback import ControlledWavPlayback
from tldw_chatbook.TTS.controlled_wav_profile import WavBookmark
from tldw_chatbook.TTS import controlled_wav_profile as profile

VALIDATION_FAILURES = []
ORIGINAL_VALIDATE = profile._validate


def observed_validation(*args):
    try:
        return ORIGINAL_VALIDATE(*args)
    except Exception as error:
        facts = {"exception": type(error).__name__}
        if args[2].stream is not None:
            from tldw_chatbook.Utils import private_paths

            info = private_paths.os.fstat(args[2].stream.fileno())
            facts.update(
                mode=oct(stat.S_IMODE(info.st_mode)),
                owner_matches=info.st_uid == private_paths.os.geteuid(),
                links=info.st_nlink,
                size=info.st_size,
            )
        VALIDATION_FAILURES.append(facts)
        raise


class Capture:
    def __init__(self, duration):
        self.duration = duration
        self.packets, self.polls = [], []
        self.config, self.metrics = {}, {}
        self.error = None
        self.ready_at = None
        self.ready = threading.Event()
        self.done = threading.Event()
        self.reader = None
        self.receiver, sender = mp.get_context("spawn").Pipe(duplex=False)
        self.sender = sender
        self.process = mp.get_context("spawn").Process(
            target=q._capture_worker, args=(sender, duration, 4800, "wasapi-packet")
        )

    def drain(self):
        frames = 0
        try:
            while True:
                message = self.receiver.recv()
                if message[0] == "ready":
                    self.ready_at, self.config = message[1:]
                    self.polls.append(self.ready_at)
                    self.ready.set()
                elif message[0] == "packets":
                    _, stamp, batch, self.metrics = message
                    self.polls.append(stamp)
                    self.packets.extend(batch)
                    frames += sum(len(packet.data) for packet in batch)
                    if frames > q.RATE * (self.duration + 2):
                        self.error = "capture_frame_bound"
                        break
                elif message[0] == "error":
                    self.error = message[1]
                    break
        except EOFError:
            pass
        finally:
            self.ready.set()
            self.done.set()

    async def start(self):
        await asyncio.to_thread(self.process.start)
        self.sender.close()
        self.reader = threading.Thread(target=self.drain, daemon=True)
        self.reader.start()
        if not await asyncio.to_thread(self.ready.wait, 15):
            raise RuntimeError("capture_ready_timeout")
        if self.error or self.ready_at is None:
            raise RuntimeError(self.error or "capture_not_ready")

    async def finish(self):
        if self.process.pid is None:
            self.sender.close()
            self.receiver.close()
            return True
        await asyncio.to_thread(self.process.join, self.duration + 2)
        if self.process.is_alive():
            self.error = self.error or "capture_timeout"
            self.process.kill()
            await asyncio.to_thread(self.process.join, 1.5)
        if self.reader is not None:
            await asyncio.to_thread(self.reader.join, 2)
        if self.process.exitcode != 0:
            self.error = self.error or "capture_process_failed"
        self.receiver.close()
        return not self.process.is_alive() and (
            self.reader is None or not self.reader.is_alive()
        )

    def quiet(self, start, end):
        return q.quiet_window(
            self.packets,
            self.polls,
            start,
            end,
            max_gap_s=min(0.15, self.config["actual_buffer_frames"] / q.RATE),
            drain_delay_s=self.config["device_period_s"],
        )


async def control(fixture, case, duration, offset, action=None):
    artifact, source, _ = fixture
    facts = ci.fixture_receipt(artifact)
    VALIDATION_FAILURES.clear()
    bookmark = WavBookmark(artifact.sha256, q.RATE, round(offset * q.RATE))
    try:
        run = await ControlledWavPlayback.prepare(artifact, bookmark)
    except profile.ControlledWavError as error:
        return {
            "case": case,
            "fixture": facts,
            "fixture_native_mode_after_retirement": ci.fixture_mode(artifact.path),
            "prepare_refusal": str(error),
            "validation_diagnostics": list(VALIDATION_FAILURES),
            "no_launch": True,
        }
    capture = Capture(max(4.0, duration - offset + 2.5))
    row = {
        "case": case,
        "fixture": facts,
        "content_frames": len(source),
        "start_frame": bookmark.frame,
    }
    row["fixture_native_mode_at_admission"] = ci.fixture_mode(artifact.path)
    host = AudioOutputOwnership()
    try:
        await capture.start()
        await asyncio.sleep(0.45)
        prelaunch = time.perf_counter()
        baseline = capture.quiet(capture.ready_at, prelaunch - 0.05)
        row["baseline_monitor"] = baseline
        if baseline["status"] != "pass":
            row["reason"] = (
                "noisy_baseline"
                if baseline["status"] == "fail"
                else "baseline_unobserved"
            )
            return row
        if not run.already_at_end:
            lease = await host.acquire(case, run.retire)
            startup = asyncio.create_task(run.start(lease))
            if action == "startup_stop":
                deadline = time.perf_counter() + 5
                while run._clock is None and time.perf_counter() < deadline:
                    await asyncio.sleep(0.001)
                launched = run._clock.launched_at if run._clock else time.perf_counter()
                await asyncio.sleep(max(0, launched + 0.020 - time.perf_counter()))
                row["stop_acknowledged"] = await host.stop(lease)
            else:
                await asyncio.shield(startup)
                if action in ("active_stop", "padding_stop"):
                    deadline = time.perf_counter() + 8
                    while time.perf_counter() < deadline:
                        observation = run.position()
                        if observation and observation.frame >= (
                            artifact.content_frames
                            if action == "padding_stop"
                            else bookmark.frame + round(0.5 * q.RATE)
                        ):
                            row["stop_observation"] = asdict(observation)
                            row["stop_acknowledged"] = await host.stop(lease)
                            break
                        if run._completion.done():
                            row["stop_trigger_missed"] = True
                            break
                        await asyncio.sleep(0.002)
            await asyncio.shield(startup)
        completion = await asyncio.wait_for(run.completion, 12)
        row["completion"] = asdict(completion)
        row["retirement_confirmed"] = await run.retire()
        row["owner_released"] = host.owner is None
        await asyncio.sleep(0.45)
    except Exception as error:
        row["error"] = type(error).__name__
    finally:
        row["final_retirement_confirmed"] = await run.retire()
        row["owner_released"] = host.owner is None
        row["capture_reaped"] = await capture.finish()
        row["capture_error"] = capture.error
        row["capture_configuration"] = capture.config
        row["capture_metrics"] = capture.metrics
        row["fixture_native_mode_after_retirement"] = ci.fixture_mode(artifact.path)
    if row.get("error") or row.get("capture_error"):
        return row
    completion = row.get("completion", {})
    evidence = completion.get("evidence", {})
    launched = evidence.get("launched_at_s")
    reaped = evidence.get("reaped_at_s")
    if launched is not None:
        packets = [packet for packet in capture.packets if packet.end_s > launched]
        match, diagnostics = q._match_packets(packets, source)
        row.update(waveform_match=match, waveform_diagnostics=diagnostics)
        if match:
            row["start_error_s"] = match["start_source_s"] - offset
            row["end_error_s"] = match["end_source_s"] - duration
        if reaped is not None:
            row["post_reap_quiet"] = capture.quiet(reaped + 0.075, reaped + 0.4)
    return row


CASES = [
    ("natural_500ms", 0.5, 0, None),
    ("natural_120ms", 0.12, 0, None),
    ("natural_2500ms", 2.5, 0, None),
    ("resume_1230ms", 2.5, 1.23, None),
    ("near_eof_140ms", 2.5, 2.36, None),
    ("near_eof_60ms", 2.5, 2.44, None),
    ("near_eof_one_frame", 2.5, 2.5 - 1 / q.RATE, None),
    ("exact_eof", 2.5, 2.5, None),
    ("startup_stop", 2.5, 0, "startup_stop"),
    ("active_stop", 2.5, 0, "active_stop"),
    ("padding_stop", 0.5, 0, "padding_stop"),
]


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--play", action="store_true")
    mode.add_argument("--quiet-preflight", action="store_true")
    parser.add_argument("--case", choices=tuple(case[0] for case in CASES))
    args = parser.parse_args()
    output, metadata = ci.new_output(args.output_dir)
    if args.quiet_preflight:
        capture = Capture(2)
        try:
            await capture.start()
            await asyncio.sleep(0.6)
            end = time.perf_counter() - 0.05
        finally:
            reaped = await capture.finish()
        receipt = {
            "baseline_monitor": capture.quiet(capture.ready_at, end),
            "capture_configuration": capture.config,
            "capture_metrics": capture.metrics,
            "capture_error": capture.error,
            "capture_reaped": reaped,
        }
        ci.save(output / "quiet-preflight.json", receipt)
        print(json.dumps(receipt), flush=True)
        return (
            0
            if reaped
            and capture.error is None
            and receipt["baseline_monitor"]["status"] == "pass"
            else 1
        )
    cases = [item for item in CASES if args.case is None or item[0] == args.case]
    fixtures = ci.Fixtures()
    result = {**metadata, "runs": []}
    profile._validate = observed_validation
    clean = False
    try:
        for case in cases:
            clean = False
            # Use a new name per case, preserving this exact object for saved Resume.
            fixture = ci.make_fixture(fixtures.directory, case[1], 41177, case=case[0])
            initial_file = fixture[0].path.stat()
            row = await control(fixture, *case)
            result["runs"].append(row)
            ci.save(output / "adapter-controls.json", result)
            if row.get("prepare_refusal"):
                clean = row["prepare_refusal"] != "pin_retirement_unconfirmed"
                return 1
            clean = (
                row.get("final_retirement_confirmed")
                and row.get("capture_reaped")
                and row.get("owner_released")
            )
            if (
                not clean
                or row.get("error")
                or row["capture_error"]
                or row["baseline_monitor"]["status"] != "pass"
            ):
                return 1
            if case[0] == "active_stop" and row["completion"]["kind"] == "stopped":
                saved = row["completion"]["bookmark"]["frame"]
                if ci.sha256(fixture[0].path) != fixture[0].sha256:
                    raise RuntimeError("saved_artifact_bytes_changed")
                clean = False
                resumed = await control(
                    fixture, "resume_saved_bookmark", case[1], saved / q.RATE
                )
                resumed["source_stop_case"] = case[0]
                resumed["same_retained_file"] = bool(
                    initial_file.st_ino
                ) and os.path.samestat(initial_file, fixture[0].path.stat())
                result["runs"].append(resumed)
                ci.save(output / "adapter-controls.json", result)
                clean = (
                    resumed.get("final_retirement_confirmed")
                    and resumed.get("capture_reaped")
                    and resumed.get("owner_released")
                )
                if (
                    not clean
                    or resumed.get("error")
                    or resumed.get("capture_error")
                    or resumed.get("baseline_monitor", {}).get("status") != "pass"
                    or not resumed["same_retained_file"]
                ):
                    return 1
    finally:
        profile._validate = ORIGINAL_VALIDATE
        if clean:
            fixtures.remove()
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(asyncio.run(main()))
