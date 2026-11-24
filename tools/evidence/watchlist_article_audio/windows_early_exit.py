"""Negative native control: a known owned process exits zero before its tail."""

import argparse
import asyncio
from pathlib import Path
import threading
import time

import windows_controlled_adapter as instrument
import controlled_instrument as ci
from grade_controlled_audio import grade_negative
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
    fixtures = None
    row = {"case": "early_zero_exit"}
    try:
        fixtures = ci.Fixtures()
        playback._spawn = early_zero_spawn
        fixture = ci.make_fixture(fixtures.directory, 2.5, 41177)
        row = await instrument.control(fixture, "early_zero_exit", 2.5, 0)
    except Exception:
        row["error"] = "acquisition_failed"
    finally:
        playback._spawn = ORIGINAL_SPAWN
        for thread in THREADS:
            try:
                await asyncio.to_thread(thread.join, 2)
            except Exception:
                row["error"] = "fault_worker_join_failed"
        row["fault"] = FAULT.copy()
    settled = not any(thread.is_alive() for thread in THREADS)
    if not settled:
        row["error"] = "fault_worker_unsettled"
    try:
        ci.save(output / "negative-control.json", {**metadata, **row})
        grade_negative(row, promoted=True)
    except (KeyError, TypeError, ValueError):
        return 1
    finally:
        if (
            fixtures is not None
            and row.get("final_retirement_confirmed") is True
            and row.get("capture_reaped") is True
            and row.get("owner_released") is True
            and FAULT.get("duplicate_handle_closed") is True
            and settled
        ):
            fixtures.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
