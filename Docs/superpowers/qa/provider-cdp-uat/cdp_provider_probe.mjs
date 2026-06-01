import { createRequire } from "module";
import { existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const PLAYWRIGHT_BUNDLE_PATH =
  "/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright";
const CHROME_EXECUTABLE = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const __dirname = dirname(fileURLToPath(import.meta.url));
const SCREENSHOT_DIR = resolve(__dirname, "screenshots");
const DEFAULT_TEXTUAL_WEB_PORT = "8877";
const DEFAULT_BROWSER_TIMEOUT_MS = 30000;
const DEFAULT_WAIT_FOR_LOG_TIMEOUT_MS = 45000;
const MAX_LOG_TAIL_BYTES = 20000;

const COMMANDS = new Set([
  "--help",
  "-h",
  "help",
  "screenshot",
  "focusTerminal",
  "typeText",
  "press",
  "readLogTail",
  "waitForLog",
]);

function printUsage() {
  console.log(`Provider CDP UAT probe

Usage:
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs --help
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs screenshot <name>
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs focusTerminal
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs typeText <text>
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs press <key>
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs readLogTail [offset]
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs waitForLog <regex> [timeoutMs] [offset]

Environment:
  TLDW_TEXTUAL_WEB_PORT  Port for the local Textual-web server.
  TLDW_QA_APP_LOG        App log path used by log commands.

Notes:
  - --help does not require a running server.
  - Browser commands connect to the local Textual-web server.
  - Log output is redacted before printing.
  - Screenshots are written under Docs/superpowers/qa/provider-cdp-uat/screenshots/.
`);
}

function parseOffset(rawOffset) {
  if (rawOffset === undefined) {
    return 0;
  }
  if (!/^\d+$/.test(rawOffset)) {
    throw new Error("Offset must be a non-negative integer.");
  }
  return Number(rawOffset);
}

function parseTimeout(rawTimeout) {
  if (rawTimeout === undefined) {
    return DEFAULT_WAIT_FOR_LOG_TIMEOUT_MS;
  }
  if (!/^\d+$/.test(rawTimeout)) {
    throw new Error("Timeout must be a non-negative integer.");
  }
  return Number(rawTimeout);
}

function assertArgCount(command, args, min, max = min) {
  if (args.length < min || args.length > max) {
    const expected = min === max ? String(min) : `${min}-${max}`;
    throw new Error(`${command} expected ${expected} argument(s), got ${args.length}.`);
  }
}

function getTextualWebPort() {
  const port = process.env.TLDW_TEXTUAL_WEB_PORT || DEFAULT_TEXTUAL_WEB_PORT;
  if (!/^\d+$/.test(port)) {
    throw new Error("TLDW_TEXTUAL_WEB_PORT must be numeric.");
  }
  const numericPort = Number(port);
  if (numericPort < 1 || numericPort > 65535) {
    throw new Error("TLDW_TEXTUAL_WEB_PORT must be between 1 and 65535.");
  }
  return port;
}

function getAppLogPath() {
  const appLogPath = process.env.TLDW_QA_APP_LOG;
  if (!appLogPath) {
    throw new Error("TLDW_QA_APP_LOG is required for log commands.");
  }
  return appLogPath;
}

function sanitizeText(text) {
  return text
    .replace(/\bBearer\s+[A-Za-z0-9._~+/=-]{10,}/gi, "Bearer [REDACTED]")
    .replace(/\b(sk-ant-[A-Za-z0-9_-]{8,})\b/g, "[REDACTED_KEY]")
    .replace(/\b(sk-[A-Za-z0-9_-]{12,})\b/g, "[REDACTED_KEY]")
    .replace(/\b(AIza[0-9A-Za-z_-]{16,})\b/g, "[REDACTED_KEY]")
    .replace(/\b([A-Za-z0-9_-]{32,})\b/g, "[REDACTED_TOKEN]")
    .replace(
      /\b(api[_-]?key|token|secret|authorization|x-api-key)(\s*[:=]\s*)(["']?)[^"',\s;]+/gi,
      "$1$2$3[REDACTED]"
    );
}

function sanitizeErrorMessage(error) {
  const message = error instanceof Error ? error.message : String(error);
  return sanitizeText(message).replace(/http:\/\/127\.0\.0\.1:\d+/g, "http://127.0.0.1:[PORT]");
}

function readLogTail(offset = 0) {
  const appLogPath = getAppLogPath();
  if (!existsSync(appLogPath)) {
    return {
      offset,
      nextOffset: offset,
      bytesRead: 0,
      textBytes: 0,
      truncated: false,
      truncatedBytes: 0,
      maxTailBytes: MAX_LOG_TAIL_BYTES,
      text: "",
    };
  }

  const size = statSync(appLogPath).size;
  const safeOffset = Math.min(offset, size);
  const rawTailBuffer = readFileSync(appLogPath).subarray(safeOffset);
  const truncated = rawTailBuffer.length > MAX_LOG_TAIL_BYTES;
  const clippedTailBuffer = truncated
    ? rawTailBuffer.subarray(rawTailBuffer.length - MAX_LOG_TAIL_BYTES)
    : rawTailBuffer;

  return {
    offset: safeOffset,
    nextOffset: size,
    bytesRead: size - safeOffset,
    textBytes: clippedTailBuffer.length,
    truncated,
    truncatedBytes: truncated ? rawTailBuffer.length - clippedTailBuffer.length : 0,
    maxTailBytes: MAX_LOG_TAIL_BYTES,
    text: sanitizeText(clippedTailBuffer.toString("utf8")),
  };
}

function readLogTextFromOffset(offset = 0) {
  const appLogPath = getAppLogPath();
  if (!existsSync(appLogPath)) {
    return { offset, nextOffset: offset, text: "" };
  }

  const size = statSync(appLogPath).size;
  const safeOffset = Math.min(offset, size);
  const text = readFileSync(appLogPath).subarray(safeOffset).toString("utf8");
  return { offset: safeOffset, nextOffset: size, text };
}

async function waitForLog(pattern, timeoutMs = DEFAULT_WAIT_FOR_LOG_TIMEOUT_MS, offset) {
  const startOffset = offset ?? (existsSync(getAppLogPath()) ? statSync(getAppLogPath()).size : 0);
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    const tail = readLogTextFromOffset(startOffset);
    pattern.lastIndex = 0;
    if (pattern.test(sanitizeText(tail.text))) {
      return { matched: true, offset: startOffset, nextOffset: tail.nextOffset };
    }
    await new Promise((resolveTimeout) => setTimeout(resolveTimeout, 250));
  }

  throw new Error("Timed out waiting for app log pattern.");
}

function compileRegex(rawPattern) {
  if (!rawPattern) {
    throw new Error("waitForLog requires a regex pattern.");
  }
  const slashPattern = rawPattern.match(/^\/(.+)\/([dgimsuvy]*)$/);
  if (slashPattern) {
    return new RegExp(slashPattern[1], slashPattern[2]);
  }
  return new RegExp(rawPattern);
}

function loadPlaywright() {
  return require(PLAYWRIGHT_BUNDLE_PATH);
}

async function settle(page, ms = 500) {
  await page.evaluate(() => {
    document.body.classList.add("-first-byte");
    for (const element of document.querySelectorAll(".intro-dialog,.closed-dialog,.shade")) {
      element.style.pointerEvents = "none";
    }
  });
  await page.waitForTimeout(ms);
}

async function withPage(operation) {
  const { chromium } = loadPlaywright();
  const browser = await chromium.launch({
    headless: true,
    executablePath: CHROME_EXECUTABLE,
  });

  try {
    const page = await browser.newPage({
      viewport: { width: 2050, height: 1240 },
      deviceScaleFactor: 1,
    });
    await page.route("https://fonts.googleapis.com/**", (route) =>
      route.fulfill({ status: 200, contentType: "text/css", body: "" })
    );
    await page.goto(`http://127.0.0.1:${getTextualWebPort()}`, {
      waitUntil: "domcontentloaded",
      timeout: DEFAULT_BROWSER_TIMEOUT_MS,
    });
    await page.waitForSelector("body", { state: "attached", timeout: DEFAULT_BROWSER_TIMEOUT_MS });
    await page.waitForSelector(".xterm-helper-textarea", {
      state: "attached",
      timeout: DEFAULT_BROWSER_TIMEOUT_MS,
    });
    await settle(page, 1000);
    return await operation(page);
  } finally {
    await browser.close();
  }
}

function normalizeScreenshotName(name) {
  if (!name || !/^[A-Za-z0-9._-]+$/.test(name)) {
    throw new Error("Screenshot name must use only letters, numbers, dots, underscores, or hyphens.");
  }
  const normalizedName = name.endsWith(".png") ? name : `${name}.png`;
  return resolve(SCREENSHOT_DIR, normalizedName);
}

async function screenshot(name) {
  const screenshotPath = normalizeScreenshotName(name);
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  await withPage(async (page) => {
    await settle(page, 500);
    await page.screenshot({ path: screenshotPath, fullPage: true });
  });
  console.log(screenshotPath);
}

async function focusTerminal() {
  await withPage(async (page) => {
    await focusTerminalOnPage(page);
  });
  console.log("focused");
}

async function focusTerminalOnPage(page) {
  const helper = page.locator(".xterm-helper-textarea").first();
  await helper.waitFor({ state: "attached", timeout: 10000 });
  await helper.focus({ timeout: 5000 });
  await page.mouse.click(305, 1170);
  await settle(page, 200);

  const focusState = await page.evaluate(() => {
    const helperElement = document.querySelector(".xterm-helper-textarea");
    const activeElement = document.activeElement;
    return {
      helperPresent: Boolean(helperElement),
      activeIsHelper: Boolean(helperElement && activeElement === helperElement),
      terminalHasFocus: Boolean(document.querySelector(".xterm.focus,.terminal.xterm.focus")),
    };
  });

  if (!focusState.helperPresent || (!focusState.activeIsHelper && !focusState.terminalHasFocus)) {
    throw new Error("Unable to focus Textual-web terminal input.");
  }
}

async function typeText(text) {
  await withPage(async (page) => {
    await focusTerminalOnPage(page);
    await page.keyboard.type(text, { delay: 5 });
  });
  console.log(`typed ${text.length} characters`);
}

async function press(key) {
  await withPage(async (page) => {
    await focusTerminalOnPage(page);
    await page.keyboard.press(key);
  });
  console.log(`pressed ${key}`);
}

async function main() {
  const [command, ...args] = process.argv.slice(2);
  if (!command || command === "--help" || command === "-h" || command === "help") {
    if (args.length > 0) {
      throw new Error(`${command || "--help"} does not accept arguments.`);
    }
    printUsage();
    return;
  }

  if (!COMMANDS.has(command)) {
    throw new Error(`Unknown command: ${command}`);
  }

  if (command === "screenshot") {
    assertArgCount(command, args, 1);
    await screenshot(args[0]);
  } else if (command === "focusTerminal") {
    assertArgCount(command, args, 0);
    await focusTerminal();
  } else if (command === "typeText") {
    assertArgCount(command, args, 1, Number.MAX_SAFE_INTEGER);
    const text = args.join(" ");
    if (!text) {
      throw new Error("typeText requires text.");
    }
    await typeText(text);
  } else if (command === "press") {
    assertArgCount(command, args, 1);
    await press(args[0]);
  } else if (command === "readLogTail") {
    assertArgCount(command, args, 0, 1);
    console.log(JSON.stringify(readLogTail(parseOffset(args[0])), null, 2));
  } else if (command === "waitForLog") {
    assertArgCount(command, args, 1, 3);
    const offset = args[2] === undefined ? undefined : parseOffset(args[2]);
    const result = await waitForLog(compileRegex(args[0]), parseTimeout(args[1]), offset);
    console.log(JSON.stringify(result, null, 2));
  }
}

main().catch((error) => {
  console.error(`error: ${sanitizeErrorMessage(error)}`);
  process.exitCode = 1;
});
