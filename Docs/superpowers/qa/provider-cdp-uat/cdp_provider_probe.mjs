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
  "inspectDom",
  "clickTextThenInspect",
  "clickTwoTextsInspect",
  "clickTwoTextsPressKeysInspect",
  "clickTextThenScreenshot",
  "clickSelectorThenInspect",
  "clickSelectorThenScreenshot",
  "clickTextPressKeysInspect",
  "clickTextPressKeysTypeInspect",
  "clickTextOffsetTypeInspect",
  "clickTextPressKeysOffsetTypeInspect",
  "clickTextPressKeysOffsetTypePressKeysInspect",
  "clickTextPressKeysClickAtTypePressKeysInspect",
  "openSettingsSaveInspect",
  "providerModelKeysInspect",
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
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs screenshot <name> [waitMs]
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs inspectDom [waitMs]
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs clickTextThenInspect <text> [waitMs]
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs clickTwoTextsInspect <firstText> <secondText> [waitMs]
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs clickTwoTextsPressKeysInspect <firstText> <secondText> <waitMs> <key...>
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs clickTextThenScreenshot <text> <name> [waitMs]
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs clickSelectorThenInspect <selector> [waitMs]
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs clickSelectorThenScreenshot <selector> <name> [waitMs]
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs clickTextPressKeysInspect <text> <waitMs> <key...>
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs clickTextPressKeysTypeInspect <text> <waitMs> <typeText> <key...>
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs clickTextOffsetTypeInspect <text> <charOffset> <waitMs> <typeText>
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs clickTextPressKeysOffsetTypeInspect <text> <offsetText> <charOffset> <waitMs> <typeText> <key...>
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs clickTextPressKeysOffsetTypePressKeysInspect <text> <offsetText> <charOffset> <waitMs> <typeText> -- <preKey...> -- <postKey...>
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs clickTextPressKeysClickAtTypePressKeysInspect <text> <x> <y> <waitMs> <typeText> -- <preKey...> -- <postKey...>
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs openSettingsSaveInspect <mode> [waitMs]
  node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs providerModelKeysInspect <waitMs> -- <providerKey...> -- <modelKey...>
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

function parseWaitMs(rawWaitMs) {
  if (rawWaitMs === undefined) {
    return 0;
  }
  if (!/^\d+$/.test(rawWaitMs)) {
    throw new Error("waitMs must be a non-negative integer.");
  }
  const waitMs = Number(rawWaitMs);
  if (waitMs > 60000) {
    throw new Error("waitMs must be 60000 or less.");
  }
  return waitMs;
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

async function collectDomInspection(page) {
  return await page.evaluate(() => {
    const rectFor = (element) => {
      const rect = element.getBoundingClientRect();
      return {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
      };
    };
    const bodyStyle = getComputedStyle(document.body);
    const terminal = document.querySelector(".terminal,.xterm");
    const terminalStyle = terminal ? getComputedStyle(terminal) : null;
    const helper = document.querySelector(".xterm-helper-textarea");
    const rows = [...document.querySelectorAll(".xterm-rows div")]
      .slice(0, 90)
      .map((row) => row.textContent || "");
    const canvases = [...document.querySelectorAll("canvas")].map((canvas) => ({
      width: canvas.width,
      height: canvas.height,
      rect: rectFor(canvas),
    }));

    return {
      title: document.title,
      url: location.href,
      bodyTextLength: document.body.innerText.length,
      bodyTextSample: document.body.innerText.slice(0, 1000),
      bodyBackground: bodyStyle.backgroundColor,
      terminalPresent: Boolean(terminal),
      terminalTextLength: terminal?.textContent?.length || 0,
      terminalTextSample: terminal?.textContent?.slice(0, 1000) || "",
      terminalBackground: terminalStyle?.backgroundColor || null,
      helperPresent: Boolean(helper),
      helperFocused: Boolean(helper && document.activeElement === helper),
      helperRect: helper ? rectFor(helper) : null,
      canvasCount: canvases.length,
      canvases,
      rows,
    };
  });
}

async function clickTerminalText(page, text, charOffset = null) {
  if (!text) {
    throw new Error("click text is required.");
  }
  const target = await page.evaluate(({ needle, offset }) => {
    const rowElements = [...document.querySelectorAll(".xterm-rows div")];
    for (const row of rowElements) {
      const rowText = row.textContent || "";
      const index = rowText.indexOf(needle);
      if (index < 0) {
        continue;
      }
      const rect = row.getBoundingClientRect();
      const columns = Math.max(rowText.length, 1);
      const cellWidth = rect.width / columns;
      const column = offset === null ? index + needle.length / 2 : index + Number(offset);
      return {
        x: rect.x + column * cellWidth,
        y: rect.y + rect.height / 2,
        rowText,
      };
    }
    return null;
  }, { needle: text, offset: charOffset });
  if (!target) {
    throw new Error(`Text not found in terminal rows: ${text}`);
  }
  await page.mouse.click(target.x, target.y);
  await settle(page, 500);
  return {
    clickedText: text,
    x: target.x,
    y: target.y,
    rowText: target.rowText,
  };
}

async function clickSelector(page, selector) {
  if (!selector || !/^[#.][A-Za-z0-9_.:-]+$/.test(selector)) {
    throw new Error("Selector must be a simple id or class selector.");
  }
  const locator = page.locator(selector).first();
  await locator.waitFor({ state: "attached", timeout: 10000 });
  const box = await locator.boundingBox();
  if (!box) {
    throw new Error(`Selector is not visible: ${selector}`);
  }
  await locator.click({ timeout: 10000 });
  await settle(page, 500);
  return {
    clickedSelector: selector,
    x: box.x + box.width / 2,
    y: box.y + box.height / 2,
    width: box.width,
    height: box.height,
  };
}

function normalizeScreenshotName(name) {
  if (!name || !/^[A-Za-z0-9._-]+$/.test(name)) {
    throw new Error("Screenshot name must use only letters, numbers, dots, underscores, or hyphens.");
  }
  const normalizedName = name.endsWith(".png") ? name : `${name}.png`;
  return resolve(SCREENSHOT_DIR, normalizedName);
}

async function screenshot(name, waitMs = 0) {
  const screenshotPath = normalizeScreenshotName(name);
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    await page.screenshot({ path: screenshotPath, fullPage: true });
  });
  console.log(screenshotPath);
}

async function inspectDom(waitMs = 0) {
  const inspection = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    return await collectDomInspection(page);
  });

  console.log(JSON.stringify(JSON.parse(sanitizeText(JSON.stringify(inspection))), null, 2));
}

async function clickTextThenInspect(text, waitMs = 0) {
  const result = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const click = await clickTerminalText(page, text);
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    return {
      click,
      inspection: await collectDomInspection(page),
    };
  });
  console.log(JSON.stringify(JSON.parse(sanitizeText(JSON.stringify(result))), null, 2));
}

async function clickTwoTextsInspect(firstText, secondText, waitMs = 0) {
  const result = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const firstClick = await clickTerminalText(page, firstText);
    await page.waitForTimeout(500);
    const secondClick = await clickTerminalText(page, secondText);
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    return {
      firstClick,
      secondClick,
      inspection: await collectDomInspection(page),
    };
  });
  console.log(JSON.stringify(JSON.parse(sanitizeText(JSON.stringify(result))), null, 2));
}

async function clickTwoTextsPressKeysInspect(firstText, secondText, waitMs, keys) {
  const result = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const firstClick = await clickTerminalText(page, firstText);
    await page.waitForTimeout(500);
    const secondClick = await clickTerminalText(page, secondText);
    for (const key of keys) {
      await page.keyboard.press(key);
      await page.waitForTimeout(300);
    }
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    return {
      firstClick,
      secondClick,
      keys,
      inspection: await collectDomInspection(page),
    };
  });
  console.log(JSON.stringify(JSON.parse(sanitizeText(JSON.stringify(result))), null, 2));
}

async function clickTextThenScreenshot(text, name, waitMs = 0) {
  const screenshotPath = normalizeScreenshotName(name);
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  const click = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const clickResult = await clickTerminalText(page, text);
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    return clickResult;
  });
  console.log(JSON.stringify({ screenshotPath, click }, null, 2));
}

async function clickSelectorThenInspect(selector, waitMs = 0) {
  const result = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const click = await clickSelector(page, selector);
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    return {
      click,
      inspection: await collectDomInspection(page),
    };
  });
  console.log(JSON.stringify(JSON.parse(sanitizeText(JSON.stringify(result))), null, 2));
}

async function clickSelectorThenScreenshot(selector, name, waitMs = 0) {
  const screenshotPath = normalizeScreenshotName(name);
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  const click = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const clickResult = await clickSelector(page, selector);
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    return clickResult;
  });
  console.log(JSON.stringify({ screenshotPath, click }, null, 2));
}

async function clickTextPressKeysInspect(text, waitMs, keys) {
  const result = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const click = await clickTerminalText(page, text);
    for (const key of keys) {
      await page.keyboard.press(key);
      await page.waitForTimeout(200);
    }
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    return {
      click,
      keys,
      inspection: await collectDomInspection(page),
    };
  });
  console.log(JSON.stringify(JSON.parse(sanitizeText(JSON.stringify(result))), null, 2));
}

async function clickTextPressKeysTypeInspect(text, waitMs, typeTextValue, keys) {
  const result = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const click = await clickTerminalText(page, text);
    for (const key of keys) {
      await page.keyboard.press(key);
      await page.waitForTimeout(200);
    }
    await page.keyboard.type(typeTextValue, { delay: 5 });
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    return {
      click,
      keys,
      typedLength: typeTextValue.length,
      inspection: await collectDomInspection(page),
    };
  });
  console.log(JSON.stringify(JSON.parse(sanitizeText(JSON.stringify(result))), null, 2));
}

async function clickTextOffsetTypeInspect(text, charOffset, waitMs, typeTextValue) {
  const result = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const click = await clickTerminalText(page, text, charOffset);
    await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A");
    await page.keyboard.type(typeTextValue, { delay: 5 });
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    return {
      click,
      charOffset,
      typedLength: typeTextValue.length,
      inspection: await collectDomInspection(page),
    };
  });
  console.log(JSON.stringify(JSON.parse(sanitizeText(JSON.stringify(result))), null, 2));
}

async function clickTextPressKeysOffsetTypeInspect(
  text,
  offsetText,
  charOffset,
  waitMs,
  typeTextValue,
  keys,
) {
  const result = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const click = await clickTerminalText(page, text);
    for (const key of keys) {
      await page.keyboard.press(key);
      await page.waitForTimeout(200);
    }
    const inputClick = await clickTerminalText(page, offsetText, charOffset);
    await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A");
    await page.keyboard.type(typeTextValue, { delay: 5 });
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    return {
      click,
      inputClick,
      keys,
      charOffset,
      typedLength: typeTextValue.length,
      inspection: await collectDomInspection(page),
    };
  });
  console.log(JSON.stringify(JSON.parse(sanitizeText(JSON.stringify(result))), null, 2));
}

async function clickTextPressKeysOffsetTypePressKeysInspect(
  text,
  offsetText,
  charOffset,
  waitMs,
  typeTextValue,
  preKeys,
  postKeys,
) {
  const result = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const click = await clickTerminalText(page, text);
    for (const key of preKeys) {
      await page.keyboard.press(key);
      await page.waitForTimeout(200);
    }
    const inputClick = await clickTerminalText(page, offsetText, charOffset);
    await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A");
    await page.keyboard.type(typeTextValue, { delay: 5 });
    await page.waitForTimeout(300);
    for (const key of postKeys) {
      await page.keyboard.press(key);
      await page.waitForTimeout(300);
    }
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    return {
      click,
      inputClick,
      preKeys,
      postKeys,
      charOffset,
      typedLength: typeTextValue.length,
      inspection: await collectDomInspection(page),
    };
  });
  console.log(JSON.stringify(JSON.parse(sanitizeText(JSON.stringify(result))), null, 2));
}

async function clickTextPressKeysClickAtTypePressKeysInspect(
  text,
  x,
  y,
  waitMs,
  typeTextValue,
  preKeys,
  postKeys,
) {
  const result = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const click = await clickTerminalText(page, text);
    for (const key of preKeys) {
      await page.keyboard.press(key);
      await page.waitForTimeout(200);
    }
    await page.mouse.click(x, y);
    await page.waitForTimeout(300);
    await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A");
    await page.keyboard.type(typeTextValue, { delay: 5 });
    await page.waitForTimeout(300);
    for (const key of postKeys) {
      await page.keyboard.press(key);
      await page.waitForTimeout(300);
    }
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    return {
      click,
      clickAt: { x, y },
      preKeys,
      postKeys,
      typedLength: typeTextValue.length,
      inspection: await collectDomInspection(page),
    };
  });
  console.log(JSON.stringify(JSON.parse(sanitizeText(JSON.stringify(result))), null, 2));
}

async function openSettingsSaveInspect(mode, waitMs = 0) {
  if (!["click", "enter", "space", "click-enter", "click-space", "double-click"].includes(mode)) {
    throw new Error("mode must be one of: click, enter, space, click-enter, click-space, double-click.");
  }
  const result = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const openClick = await clickTerminalText(page, "Configure");
    await page.waitForTimeout(500);

    const saveClick = await clickTerminalText(page, "Save");
    if (mode === "enter" || mode === "click-enter") {
      await page.keyboard.press("Enter");
    } else if (mode === "space" || mode === "click-space") {
      await page.keyboard.press("Space");
    } else if (mode === "double-click") {
      await page.mouse.click(saveClick.x, saveClick.y);
    }

    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    const rows = (await collectDomInspection(page)).rows;
    return {
      mode,
      openClick,
      saveClick,
      modalOpen: rows.join("\n").includes("Console Settings"),
      rows,
    };
  });
  console.log(JSON.stringify(JSON.parse(sanitizeText(JSON.stringify(result))), null, 2));
}

async function providerModelKeysInspect(waitMs, providerKeys, modelKeys) {
  const result = await withPage(async (page) => {
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    const openClick = await clickTerminalText(page, "Configure");
    await page.waitForTimeout(500);
    const providerClick = await clickTerminalText(page, "Provider        ▊");
    for (const key of providerKeys) {
      await page.keyboard.press(key);
      await page.waitForTimeout(300);
    }
    const modelClick = await clickTerminalText(page, "Model           ▊");
    for (const key of modelKeys) {
      await page.keyboard.press(key);
      await page.waitForTimeout(300);
    }
    if (waitMs > 0) {
      await page.waitForTimeout(waitMs);
    }
    await settle(page, 500);
    return {
      openClick,
      providerClick,
      providerKeys,
      modelClick,
      modelKeys,
      inspection: await collectDomInspection(page),
    };
  });
  console.log(JSON.stringify(JSON.parse(sanitizeText(JSON.stringify(result))), null, 2));
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
    assertArgCount(command, args, 1, 2);
    await screenshot(args[0], parseWaitMs(args[1]));
  } else if (command === "inspectDom") {
    assertArgCount(command, args, 0, 1);
    await inspectDom(parseWaitMs(args[0]));
  } else if (command === "clickTextThenInspect") {
    assertArgCount(command, args, 1, 2);
    await clickTextThenInspect(args[0], parseWaitMs(args[1]));
  } else if (command === "clickTwoTextsInspect") {
    assertArgCount(command, args, 2, 3);
    await clickTwoTextsInspect(args[0], args[1], parseWaitMs(args[2]));
  } else if (command === "clickTwoTextsPressKeysInspect") {
    assertArgCount(command, args, 4, Number.MAX_SAFE_INTEGER);
    await clickTwoTextsPressKeysInspect(args[0], args[1], parseWaitMs(args[2]), args.slice(3));
  } else if (command === "clickTextThenScreenshot") {
    assertArgCount(command, args, 2, 3);
    await clickTextThenScreenshot(args[0], args[1], parseWaitMs(args[2]));
  } else if (command === "clickSelectorThenInspect") {
    assertArgCount(command, args, 1, 2);
    await clickSelectorThenInspect(args[0], parseWaitMs(args[1]));
  } else if (command === "clickSelectorThenScreenshot") {
    assertArgCount(command, args, 2, 3);
    await clickSelectorThenScreenshot(args[0], args[1], parseWaitMs(args[2]));
  } else if (command === "clickTextPressKeysInspect") {
    assertArgCount(command, args, 3, Number.MAX_SAFE_INTEGER);
    await clickTextPressKeysInspect(args[0], parseWaitMs(args[1]), args.slice(2));
  } else if (command === "clickTextPressKeysTypeInspect") {
    assertArgCount(command, args, 4, Number.MAX_SAFE_INTEGER);
    await clickTextPressKeysTypeInspect(args[0], parseWaitMs(args[1]), args[2], args.slice(3));
  } else if (command === "clickTextOffsetTypeInspect") {
    assertArgCount(command, args, 4);
    await clickTextOffsetTypeInspect(args[0], parseOffset(args[1]), parseWaitMs(args[2]), args[3]);
  } else if (command === "clickTextPressKeysOffsetTypeInspect") {
    assertArgCount(command, args, 6, Number.MAX_SAFE_INTEGER);
    await clickTextPressKeysOffsetTypeInspect(
      args[0],
      args[1],
      parseOffset(args[2]),
      parseWaitMs(args[3]),
      args[4],
      args.slice(5),
    );
  } else if (command === "clickTextPressKeysOffsetTypePressKeysInspect") {
    assertArgCount(command, args, 7, Number.MAX_SAFE_INTEGER);
    const firstSeparator = args.indexOf("--", 5);
    const secondSeparator = firstSeparator < 0 ? -1 : args.indexOf("--", firstSeparator + 1);
    if (firstSeparator < 0 || secondSeparator < 0) {
      throw new Error("Expected two -- separators for pre and post key lists.");
    }
    await clickTextPressKeysOffsetTypePressKeysInspect(
      args[0],
      args[1],
      parseOffset(args[2]),
      parseWaitMs(args[3]),
      args[4],
      args.slice(firstSeparator + 1, secondSeparator),
      args.slice(secondSeparator + 1),
    );
  } else if (command === "clickTextPressKeysClickAtTypePressKeysInspect") {
    assertArgCount(command, args, 8, Number.MAX_SAFE_INTEGER);
    const firstSeparator = args.indexOf("--", 5);
    const secondSeparator = firstSeparator < 0 ? -1 : args.indexOf("--", firstSeparator + 1);
    if (firstSeparator < 0 || secondSeparator < 0) {
      throw new Error("Expected two -- separators for pre and post key lists.");
    }
    await clickTextPressKeysClickAtTypePressKeysInspect(
      args[0],
      Number(args[1]),
      Number(args[2]),
      parseWaitMs(args[3]),
      args[4],
      args.slice(firstSeparator + 1, secondSeparator),
      args.slice(secondSeparator + 1),
    );
  } else if (command === "openSettingsSaveInspect") {
    assertArgCount(command, args, 1, 2);
    await openSettingsSaveInspect(args[0], parseWaitMs(args[1]));
  } else if (command === "providerModelKeysInspect") {
    assertArgCount(command, args, 4, Number.MAX_SAFE_INTEGER);
    const firstSeparator = args.indexOf("--", 1);
    const secondSeparator = firstSeparator < 0 ? -1 : args.indexOf("--", firstSeparator + 1);
    if (firstSeparator < 0 || secondSeparator < 0) {
      throw new Error("Expected two -- separators for provider and model key lists.");
    }
    await providerModelKeysInspect(
      parseWaitMs(args[0]),
      args.slice(firstSeparator + 1, secondSeparator),
      args.slice(secondSeparator + 1),
    );
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
