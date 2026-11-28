"""PR3023 CDP capture preparation; final source/profile bindings are external.

Records browser dispatch and websocket arrival clocks, not native OS paint.
Drafts use atomic CDP text insertion; no clipboard access or per-character typing.
No app, provider, credential, or network action occurs until explicitly run.
"""
import argparse
import asyncio
import base64
import codecs
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import pyte
from playwright.async_api import async_playwright

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--width", required=True, type=int)
parser.add_argument("--height", required=True, type=int)
args = parser.parse_args()
assert args.width > 0 and args.height > 0
OUT = args.out.resolve()
OUT.mkdir(mode=0o700, exist_ok=False)

def stamp():
    return {"host_monotonic_ns": time.monotonic_ns(), "host_wall_ns": time.time_ns()}

async def main():
    async with async_playwright() as pw:
        owner = context = page = screen = None
        frames, sizes, actions, sent_frames, handshakes, clocks, errors = [], [], [], [], [], [], []
        try:
            owner = await pw.chromium.launch(headless=True)
            browser = owner
            context = await browser.new_context(viewport={"width": args.width, "height": args.height}, device_scale_factor=1)
            page = await context.new_page()
            cdp = await context.new_cdp_session(page)
            screen = pyte.Screen(80, 24)
            stream = pyte.Stream(screen)
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            def received(event):
                arrival = stamp()
                response = event["response"]
                if response["opcode"] == 2:
                    data = base64.b64decode(response["payloadData"])
                    frames.append({"time": event["timestamp"], "arrival": arrival, "request_id": event["requestId"], "bytes_base64": response["payloadData"]})
                    stream.feed(decoder.decode(data))
            def sent(event):
                arrival = stamp()
                response = event["response"]
                sent_frames.append({"time": event["timestamp"], "arrival": arrival, "request_id": event["requestId"], "opcode": response["opcode"], "payload_characters": len(response["payloadData"])})
                if response["opcode"] != 1:
                    return
                payload = json.loads(response["payloadData"])
                if isinstance(payload, list) and payload[0] == "resize":
                    size = payload[1]
                    screen.resize(size["height"], size["width"])
                    sizes.append({"time": event["timestamp"], "arrival": arrival, "request_id": event["requestId"], **size})
            def handshake(event):
                arrival = stamp()
                handshakes.append({"time": event["timestamp"], "wall_time_seconds": event["wallTime"], "arrival": arrival, "request_id": event["requestId"]})
            cdp.on("Network.webSocketWillSendHandshakeRequest", handshake)
            cdp.on("Network.webSocketFrameReceived", received)
            cdp.on("Network.webSocketFrameSent", sent)
            await cdp.send("Network.enable")
            await cdp.send("Performance.enable")
            async def calibration():
                before = stamp()
                metrics = await cdp.send("Performance.getMetrics")
                after = stamp()
                timestamp = next(row["value"] for row in metrics["metrics"] if row["name"] == "Timestamp")
                assert isinstance(timestamp, (int, float)) and timestamp >= 0
                return {"host_before": before, "cdp_timestamp_seconds": timestamp, "host_after": after}
            clocks.append(await calibration())
            print(json.dumps({"ready": True, "cdp": await cdp.send("Browser.getVersion"), "clock": clocks[0]}), flush=True)
            while True:
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    break
                command = json.loads(line)
                action = command["action"]
                command = dict(command)
                command["received"] = stamp()
                actions.append(command)
                try:
                    if action == "navigate":
                        screen.reset()
                        frames.clear()
                        sizes.clear()
                        sent_frames.clear()
                        handshakes.clear()
                        decoder.reset()
                        target = urlsplit(command["url"])
                        assert target.scheme == "http" and target.hostname == "127.0.0.1" and target.port is not None
                        assert not target.username and not target.password
                        await page.goto(command["url"])
                    elif action == "resize":
                        await page.set_viewport_size({"width": command["width"], "height": command["height"]})
                    elif action == "key":
                        command["pre_input"] = stamp()
                        await page.keyboard.press(command["key"])
                        command["dispatch_complete"] = stamp()
                    elif action == "type":
                        await page.locator(".xterm-helper-textarea").focus()
                        command["pre_input"] = stamp()
                        command["dispatch_method"] = "Input.insertText"
                        await page.keyboard.insert_text(command["text"])
                        command["dispatch_complete"] = stamp()
                    elif action == "click_text":
                        text = command["text"]
                        matches = [(row, value.index(text)) for row, value in enumerate(screen.display) if text in value]
                        assert len(matches) == 1, {"text": text, "matches": matches}
                        row, col = matches[0]
                        box = await page.locator(".xterm-screen").bounding_box()
                        assert box and screen.columns and screen.lines
                        x = box["x"] + (col + len(text) / 2) * box["width"] / screen.columns
                        y = box["y"] + (row + .5) * box["height"] / screen.lines
                        command["pre_input"] = stamp()
                        for kind in ("mousePressed", "mouseReleased"):
                            await cdp.send("Input.dispatchMouseEvent", {"type": kind, "x": x, "y": y, "button": "left", "clickCount": 1})
                        command["dispatch_complete"] = stamp()
                    elif action == "wait_text":
                        deadline = asyncio.get_running_loop().time() + command.get("seconds", 15)
                        while command["text"] not in "\n".join(screen.display):
                            assert asyncio.get_running_loop().time() < deadline, "text did not appear"
                            await asyncio.sleep(.05)
                    elif action == "wait_occurrences":
                        deadline = asyncio.get_running_loop().time() + command.get("seconds", 20)
                        while "\n".join(screen.display).count(command["text"]) < command["count"]:
                            assert asyncio.get_running_loop().time() < deadline, "streamed occurrences did not appear"
                            await asyncio.sleep(.025)
                    elif action == "wait_http_send":
                        path = Path(command["path"])
                        assert str(path).startswith("/private/tmp/pr3023-live-")
                        deadline = asyncio.get_running_loop().time() + command.get("seconds", 10)
                        while True:
                            records = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
                            if any(x["event"] == "send" and x["request"] == command["request"] for x in records):
                                break
                            assert asyncio.get_running_loop().time() < deadline, "actual test HTTP send was not observed"
                            await asyncio.sleep(.01)
                    elif action == "capture":
                        name = command["name"]
                        assert name.replace("-", "").replace("_", "").isalnum()
                        clocks.append(await calibration())
                        screenshot = await cdp.send("Page.captureScreenshot", {"format": "png"})
                        (OUT / (name + ".png")).write_bytes(base64.b64decode(screenshot["data"]))
                        (OUT / (name + ".txt")).write_text("\n".join(screen.display), encoding="utf-8")
                        (OUT / (name + ".json")).write_text(json.dumps({"url": page.url, "sizes": sizes, "columns": screen.columns, "rows": screen.lines, "frames": frames, "actions": actions, "clock_calibrations": clocks, "sent_frames": sent_frames, "handshakes": handshakes}, indent=2), encoding="utf-8")
                    elif action == "close":
                        break
                    else:
                        raise ValueError("unknown action")
                    command["complete"] = stamp()
                    print(json.dumps({"action": action, "ok": True, "dimensions": [screen.columns, screen.lines], "name": command.get("name"), "pre_input": command.get("pre_input"), "dispatch_complete": command.get("dispatch_complete"), "complete": command["complete"]}), flush=True)
                except Exception as error:
                    command["failed"] = stamp()
                    command["error_type"] = type(error).__name__
                    print(json.dumps({"action": action, "ok": False, "error": type(error).__name__ + ": " + str(error)}), flush=True)
        except BaseException as error:
            errors.append(error)
        finally:
            try:
                (OUT / "session.json").write_text(json.dumps({"url": page.url if page is not None else None, "columns": screen.columns if screen is not None else None, "rows": screen.lines if screen is not None else None, "frames": frames, "sizes": sizes, "actions": actions, "sent_frames": sent_frames, "handshakes": handshakes, "clock_calibrations": clocks, "operation_errors": [type(error).__name__ for error in errors]}, indent=2), encoding="utf-8")
            except BaseException as error:
                errors.append(error)
            for resource in (context, owner):
                if resource is None:
                    continue
                try:
                    await resource.close()
                except BaseException as error:
                    errors.append(error)
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise BaseExceptionGroup("CDP operation and owned cleanup failures", errors)

asyncio.run(main())
