"""Negative native control: a known owned process exits zero before its tail."""

import argparse
import asyncio
from pathlib import Path
import threading
import time

import windows_controlled_adapter as instrument
import controlled_instrument as ci
from tldw_chatbook.TTS import controlled_wav_playback as playback

FAULT = {}
THREADS = []
ORIGINAL_SPAWN = playback._spawn


def early_zero_spawn(*args, **kwargs):
    import _winapi

    child = ORIGINAL_SPAWN(*args, **kwargs)
    # Return the owned child even if the measuring fault cannot be installed.
    try:
        current = _winapi.GetCurrentProcess()
        owned = _winapi.DuplicateHandle(
            current,
            child.info.process,
            current,
            0,
            False,
            _winapi.DUPLICATE_SAME_ACCESS,
        )
    except OSError as error:
        FAULT["error"] = type(error).__name__
        return child

    def fault():
        try:
            time.sleep(0.3)
            _winapi.TerminateProcess(owned, 0)
            FAULT.update(applied=True, at_s=time.perf_counter(), requested_exit_code=0)
        except BaseException as error:
            FAULT["error"] = type(error).__name__
        finally:
            _winapi.CloseHandle(owned)
            FAULT["duplicate_handle_closed"] = True

    try:
        thread = threading.Thread(target=fault, daemon=True)
        thread.start()
        THREADS.append(thread)
    except Exception as error:
        _winapi.CloseHandle(owned)
        FAULT.update(error=type(error).__name__, duplicate_handle_closed=True)
    return child


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--play", action="store_true", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output, metadata = ci.new_output(args.output_dir)
    fixtures = ci.Fixtures()
    playback._spawn = early_zero_spawn
    row = None
    try:
        fixture = ci.make_fixture(fixtures.directory, 2.5, 41177)
        row = await instrument.control(fixture, "early_zero_exit", 2.5, 0)
        for thread in THREADS:
            await asyncio.to_thread(thread.join, 2)
        row["fault"] = FAULT.copy()
        ci.save(output / "negative-control.json", {**metadata, **row})
    finally:
        playback._spawn = ORIGINAL_SPAWN
        if (
            row
            and row.get("final_retirement_confirmed")
            and row.get("capture_reaped")
            and row.get("owner_released")
            and FAULT.get("duplicate_handle_closed")
            and not any(thread.is_alive() for thread in THREADS)
        ):
            fixtures.remove()


if __name__ == "__main__":
    asyncio.run(main())
