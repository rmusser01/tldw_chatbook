import { createRequire } from "module";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const PLAYWRIGHT_BUNDLE_PATH =
  "/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright";
const CHROME_EXECUTABLE = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const __dirname = dirname(fileURLToPath(import.meta.url));
const SCREENSHOT_DIR = resolve(__dirname, "screenshots");
const RESULTS_PATH = resolve(__dirname, "provider-sweep-results.json");
const INVENTORY_PATH = resolve(__dirname, "provider-inventory.json");
const DEFAULT_TEXTUAL_WEB_PORT = "8877";
const DEFAULT_BROWSER_TIMEOUT_MS = 30000;
const DEFAULT_STARTUP_WAIT_MS = 6000;
const DEFAULT_REPLY_TIMEOUT_MS = 120000;
const DROPDOWN_KEY_DELAY_MS = 100;

const PROVIDER_ORDER = [
  "anthropic",
  "aphrodite",
  "cohere",
  "custom",
  "custom_2",
  "deepseek",
  "google",
  "groq",
  "huggingface",
  "koboldcpp",
  "llama_cpp",
  "local_llamacpp",
  "local_llamafile",
  "local_llm",
  "local_mlx_lm",
  "local_ollama",
  "local_onnx",
  "local_transformers",
  "local_vllm",
  "mistral",
  "mistralai",
  "moonshot",
  "ollama",
  "oobabooga",
  "openai",
  "openrouter",
  "tabbyapi",
  "vllm",
  "zai",
];

function printUsage() {
  console.log(`Provider CDP UAT sweep

Usage:
  node Docs/superpowers/qa/provider-cdp-uat/run_provider_cdp_sweep.mjs [--provider <key>] [--reply-timeout-ms <ms>]

Environment:
  TLDW_TEXTUAL_WEB_PORT  Port for the local Textual-web server.

Notes:
  - Uses provider-inventory.json and attempts rows marked pending_cdp.
  - Writes redacted results to Docs/superpowers/qa/provider-cdp-uat/provider-sweep-results.json.
  - Screenshots are written under Docs/superpowers/qa/provider-cdp-uat/screenshots/.
`);
}

function parseArgs(argv) {
  const args = {
    provider: null,
    replyTimeoutMs: DEFAULT_REPLY_TIMEOUT_MS,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      args.help = true;
      continue;
    }
    if (arg === "--provider") {
      args.provider = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === "--reply-timeout-ms") {
      const rawTimeout = argv[index + 1];
      if (!/^\d+$/.test(rawTimeout || "")) {
        throw new Error("--reply-timeout-ms must be numeric.");
      }
      args.replyTimeoutMs = Number(rawTimeout);
      index += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function getTextualWebPort() {
  const port = process.env.TLDW_TEXTUAL_WEB_PORT || DEFAULT_TEXTUAL_WEB_PORT;
  if (!/^\d+$/.test(port)) {
    throw new Error("TLDW_TEXTUAL_WEB_PORT must be numeric.");
  }
  return port;
}

function sanitizeText(text) {
  return String(text || "")
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

function loadInventory() {
  if (!existsSync(INVENTORY_PATH)) {
    throw new Error(`Missing inventory: ${INVENTORY_PATH}`);
  }
  const payload = JSON.parse(readFileSync(INVENTORY_PATH, "utf8"));
  return payload.providers.filter((row) => row.initial_status === "pending_cdp");
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

async function terminalRows(page) {
  return await page.evaluate(() =>
    [...document.querySelectorAll(".xterm-rows div")].map((row) => row.textContent || "")
  );
}

function textFromRows(rows) {
  return rows.join("\n");
}

async function terminalText(page) {
  return textFromRows(await terminalRows(page));
}

async function visibleText(page) {
  return await page.evaluate(() => document.body.innerText || "");
}

async function modalIsOpen(page) {
  const [bodyText, rowsText] = await Promise.all([visibleText(page), terminalText(page)]);
  return bodyText.includes("Console Settings") || rowsText.includes("Console Settings");
}

async function clickTerminalText(page, text, charOffset = null) {
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
  await settle(page, 300);
  return target;
}

async function openPage(browser) {
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
  await page.waitForSelector(".xterm-helper-textarea", {
    state: "attached",
    timeout: DEFAULT_BROWSER_TIMEOUT_MS,
  });
  await settle(page, DEFAULT_STARTUP_WAIT_MS);
  return page;
}

async function selectProvider(page, provider) {
  const providerIndex = PROVIDER_ORDER.indexOf(provider);
  if (providerIndex < 0) {
    throw new Error(`Provider is not in the Console provider order: ${provider}`);
  }
  await clickTerminalText(page, "Configure");
  await page.waitForTimeout(300);
  await page.keyboard.press("Enter");
  await page.waitForTimeout(300);
  await page.keyboard.press("Home");
  await page.waitForTimeout(300);
  for (let index = 0; index < providerIndex; index += 1) {
    await page.keyboard.press("ArrowDown");
    await page.waitForTimeout(DROPDOWN_KEY_DELAY_MS);
  }
  await page.keyboard.press("Enter");
  await settle(page, 500);
}

async function selectModel(page, model) {
  if (!model) {
    return;
  }
  await page.keyboard.press("Tab");
  await page.waitForTimeout(300);
  await page.keyboard.press("Enter");
  await page.waitForTimeout(300);
  await page.keyboard.press("Home");
  await page.waitForTimeout(150);
  await page.keyboard.press("Enter");
  await settle(page, 500);
}

async function saveSettings(page) {
  await clickTerminalText(page, "Save");
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    await page.waitForTimeout(500);
    await settle(page, 100);
    if (!(await modalIsOpen(page))) {
      return;
    }
  }
  throw new Error("Console Settings modal did not close after Save.");
}

async function focusComposer(page) {
  if (await modalIsOpen(page)) {
    throw new Error("Cannot focus composer while Console Settings modal is open.");
  }
  await clickTerminalText(page, "Composer:", 16);
  await page.waitForTimeout(200);
}

function assistantMessageCount(text) {
  return (text.match(/\bAssistant(?:\s|\[|$)/g) || []).length;
}

function responseComplete(text) {
  return /Run:\s+Response complete\./i.test(text);
}

function failureCopy(text) {
  const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
  return sanitizeText(
    lines
      .filter((line) =>
        /Provider stream failed|Provider blocked|Console send blocked|Missing key|Model is required|Traceback|Error/i.test(line)
      )
      .slice(-6)
      .join(" | ")
  );
}

function classifyFailure(text, phase, timedOut = false) {
  const lower = text.toLowerCase();
  if (timedOut) {
    return {
      status: "fail_external",
      reason: failureCopy(text) || `provider_timeout_${phase}`,
    };
  }
  if (/console send blocked|provider blocked|model is required|select a model|not available in console|traceback/.test(lower)) {
    return {
      status: "fail_chatbook",
      reason: failureCopy(text) || `chatbook_${phase}_failure`,
    };
  }
  if (/provider stream failed|unauthorized|forbidden|invalid api|invalid key|quota|rate limit|insufficient|model.*not.*found|not found|timeout|timed out|http status|error code|apierror|bad request/.test(lower)) {
    return {
      status: "fail_external",
      reason: failureCopy(text) || `external_${phase}_failure`,
    };
  }
  return {
    status: "fail_chatbook",
    reason: failureCopy(text) || `no_completed_assistant_reply_${phase}`,
  };
}

async function waitForAssistantCount(page, expectedCount, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastRows = [];
  while (Date.now() < deadline) {
    lastRows = await terminalRows(page);
    const text = textFromRows(lastRows);
    const count = assistantMessageCount(text);
    if (count >= expectedCount && responseComplete(text) && !/\[streaming\]/i.test(text)) {
      return {
        rows: lastRows,
        text,
        count,
      };
    }
    if (/Provider stream failed|Provider blocked|Console send blocked|Model is required|Traceback/i.test(text)) {
      return {
        rows: lastRows,
        text,
        count,
      };
    }
    await page.waitForTimeout(1000);
  }
  const text = textFromRows(lastRows);
  return {
    rows: lastRows,
    text,
    count: assistantMessageCount(text),
    timedOut: true,
  };
}

async function sendTurn(page, provider, turnNumber, expectedAssistantCount, timeoutMs) {
  const message = `Provider UAT ${provider} turn ${turnNumber}: reply with one short sentence.`;
  await focusComposer(page);
  await page.keyboard.type(message, { delay: 2 });
  await page.waitForTimeout(200);
  const preSubmitText = await terminalText(page);
  if (!preSubmitText.includes(message)) {
    throw new Error(`Composer did not receive typed message for ${provider} turn ${turnNumber}.`);
  }
  await page.keyboard.press("Enter");
  return await waitForAssistantCount(page, expectedAssistantCount, timeoutMs);
}

async function screenshot(page, name) {
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  const path = resolve(SCREENSHOT_DIR, `${name}.png`);
  await settle(page, 500);
  await page.screenshot({ path, fullPage: true });
  return path;
}

async function runProvider(browser, row, replyTimeoutMs) {
  const provider = row.display_key;
  const result = {
    provider,
    execution_key: row.execution_key,
    model: row.model,
    status: "fail_chatbook",
    reason: "",
    assistant_replies: 0,
    screenshot: "",
  };

  const page = await openPage(browser);
  try {
    await selectProvider(page, provider);
    await selectModel(page, row.model);
    await saveSettings(page);

    let text = await terminalText(page);
    if (await modalIsOpen(page)) {
      result.status = "fail_chatbook";
      result.reason = "Console Settings modal remained open after save.";
      result.screenshot = await screenshot(page, `${provider}-settings-still-open`);
      return result;
    }
    if (!text.includes(`Provider: ${provider}`) || !text.includes(`Model: ${row.model}`)) {
      result.status = "fail_chatbook";
      result.reason = "Console settings summary did not show selected provider/model after save.";
      result.screenshot = await screenshot(page, `${provider}-selection-failed`);
      return result;
    }

    const first = await sendTurn(page, provider, 1, 1, replyTimeoutMs);
    if (first.count < 1 || first.timedOut || /Provider stream failed|Provider blocked|Console send blocked/i.test(first.text)) {
      const classified = classifyFailure(first.text, "turn_1", Boolean(first.timedOut));
      result.status = classified.status;
      result.reason = classified.reason;
      result.assistant_replies = first.count;
      result.screenshot = await screenshot(page, `${provider}-turn-1-${result.status}`);
      return result;
    }

    const second = await sendTurn(page, provider, 2, 2, replyTimeoutMs);
    result.assistant_replies = second.count;
    if (second.count >= 2 && !second.timedOut && !/Provider stream failed|Provider blocked|Console send blocked/i.test(second.text)) {
      result.status = "success";
      result.reason = "Second assistant reply completed.";
    } else {
      const classified = classifyFailure(second.text, "turn_2", Boolean(second.timedOut));
      result.status = classified.status;
      result.reason = classified.reason;
    }
    result.screenshot = await screenshot(page, `${provider}-${result.status}`);
    return result;
  } catch (error) {
    result.status = "fail_chatbook";
    result.reason = sanitizeErrorMessage(error);
    try {
      result.screenshot = await screenshot(page, `${provider}-exception`);
    } catch {
      result.screenshot = "";
    }
    return result;
  } finally {
    await page.close();
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printUsage();
    return;
  }

  let rows = loadInventory();
  if (args.provider) {
    rows = rows.filter((row) => row.display_key === args.provider || row.execution_key === args.provider);
    if (rows.length === 0) {
      throw new Error(`No pending_cdp inventory row matched provider: ${args.provider}`);
    }
  }

  const { chromium } = loadPlaywright();
  const browser = await chromium.launch({
    headless: true,
    executablePath: CHROME_EXECUTABLE,
  });
  const results = [];
  try {
    for (const row of rows) {
      const result = await runProvider(browser, row, args.replyTimeoutMs);
      results.push(result);
      console.log(JSON.stringify(result));
    }
  } finally {
    await browser.close();
  }

  const payload = {
    generated_at: new Date().toISOString(),
    textual_web_port: Number(getTextualWebPort()),
    attempted: results.length,
    status_counts: results.reduce((counts, result) => {
      counts[result.status] = (counts[result.status] || 0) + 1;
      return counts;
    }, {}),
    results,
  };
  writeFileSync(RESULTS_PATH, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  console.log(RESULTS_PATH);
}

main().catch((error) => {
  console.error(`error: ${sanitizeErrorMessage(error)}`);
  process.exitCode = 1;
});
