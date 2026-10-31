"""Bounded, storage-free comparison of timed versus GIL-held stack dumps."""

import faulthandler
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


def child(mode):
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            # Short-lived code/frame chains, with no application/native storage.
            exec(compile("sum(range(20))", "<ephemeral-frame>", "exec"))

    workers = [threading.Thread(target=churn) for _ in range(4)]
    for worker in workers:
        worker.start()
    try:
        if mode == "timer":
            faulthandler.dump_traceback_later(0.001, repeat=True)
            time.sleep(4)
        else:
            until = time.monotonic() + 4
            while time.monotonic() < until:
                faulthandler.dump_traceback()
                time.sleep(0.001)
    finally:
        faulthandler.cancel_dump_traceback_later()
        stop.set()
        for worker in workers:
            worker.join()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
    else:
        out = Path(__file__).parent
        results = []
        for repetition in range(3):
            for mode in ("timer", "synchronous"):
                target = out / f"traceback-stress-{mode}-{repetition}.txt"
                with target.open("wb") as log:
                    result = subprocess.run(
                        [sys.executable, __file__, mode],
                        stdout=log,
                        stderr=log,
                        timeout=30,
                    )
                results.append(
                    {
                        "mode": mode,
                        "repeat": repetition,
                        "exit_code": result.returncode,
                        "log_bytes": target.stat().st_size,
                    }
                )
                print(results[-1], flush=True)
        (out / "traceback-stress.json").write_text(
            json.dumps({"python": sys.version, "runs": results}, indent=2),
            encoding="utf-8",
        )
