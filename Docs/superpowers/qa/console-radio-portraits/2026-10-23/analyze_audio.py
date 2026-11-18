"""Score exact source transitions against native DAC mapping and terminal flushes."""

import json
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
with wave.open(str(ROOT / "calibration.wav"), "rb") as source:
    rate = source.getframerate()
    pcm = source.readframes(source.getnframes())


def at_dac(source_frame, *, prefix_bytes, callbacks, offset):
    position = prefix_bytes // 2 + source_frame
    start = 0
    for row in callbacks:
        if position < start + row["frames"]:
            return row["dac"] + offset + (position - start) / rate
        start += row["frames"]
    raise AssertionError("Source transition absent from native output")


for route in ("pcm", "wav"):
    data = json.loads((ROOT / f"native-audio-{route}-observations.json").read_text())
    with wave.open(str(ROOT / f"{route}-device-submitted.wav"), "rb") as submitted:
        output = submitted.readframes(submitted.getnframes())
    prefix_bytes = output.find(pcm)
    assert prefix_bytes >= 0 and prefix_bytes % 2 == 0, "Incomplete device submission"
    offset = data["observer_clock_offset"]
    assert offset is not None
    callbacks = data["callbacks"]

    changes = []
    prior = False
    for row in data["terminal_writes"]:
        speaking = row["color"] != [255, 0, 0]
        if speaking != prior:
            changes.append(row)
            prior = speaking
    expected = [
        (0.2, "onset", [0, 0, 255]),
        (0.8, "nonzero pause", [255, 0, 0]),
        (1.15, "onset", [0, 0, 255]),
        (1.75, "silence", [255, 0, 0]),
    ]
    assert len(changes) == len(expected), changes
    measurements = []
    for row, (seconds, event, color) in zip(changes, expected, strict=True):
        assert (row["color"] == [255, 0, 0]) == (color == [255, 0, 0])
        dac = at_dac(
            round(seconds * rate),
            prefix_bytes=prefix_bytes,
            callbacks=callbacks,
            offset=offset,
        )
        delay = (row["at"] - dac) * 1000
        measurements.append(
            {
                "event": event,
                "dac_monotonic": dac,
                "terminal_flush_monotonic": row["at"],
                "delay_ms": round(delay, 3),
            }
        )
        assert 0 <= delay <= 100, (route, event, delay)
    source_end = (
        at_dac(
            len(pcm) // 2 - 1,
            prefix_bytes=prefix_bytes,
            callbacks=callbacks,
            offset=offset,
        )
        + 1 / rate
    )
    finished = next(
        event["at"] for event in data["events"] if event["type"] == "NativeFinished"
    )
    completed = next(
        event["at"]
        for event in data["events"]
        if event["type"] == ("SinkDrained" if route == "pcm" else "WavHandlerReturned")
    )
    assert completed >= source_end and completed >= finished
    summary = {
        "route": route,
        "output_device": data["output_device"],
        "sample_rate": rate,
        "driver_latency_ms": data["driver_latency_seconds"] * 1000,
        "complete_waveform_submitted": True,
        "native_finished_after_last_scheduled_sample_ms": (finished - source_end)
        * 1000,
        "completion_after_native_finished_ms": (completed - finished) * 1000,
        "observer_clock_offset": offset,
        "maximum_callback_clock_variation_ms": max(
            abs(row["at"] - row["current"] - offset) for row in callbacks
        )
        * 1000,
        "transitions": measurements,
        "physical_loopback": False,
        "measurement": "Exact source sample boundaries mapped by production observer to monotonic; terminal flushes with actual avatar pixels. No photon/acoustic recording.",
    }
    (ROOT / f"native-audio-{route}-summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(route, [row["delay_ms"] for row in measurements])

external = json.loads((ROOT / "native-audio-external-observations.json").read_text())
assert not external["callbacks"] and external["terminal_writes"]
assert all(row["color"] == [255, 0, 0] for row in external["terminal_writes"])
assert any(event["type"] == "playing" for event in external["events"])
assert any(event["type"] == "stopped" for event in external["events"])
print("afplay: real status, no measured amplitude or mouth movement")
