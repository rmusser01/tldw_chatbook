import { createRequire } from "module";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { chromium } = require("/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");

const __dirname = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(__dirname, "screenshots");
const port = process.env.TLDW_TEXTUAL_WEB_PORT || "8877";
const appLogPath = process.env.TLDW_QA_APP_LOG || "/private/tmp/tldw-chatbook-console-native-qa-home/.local/share/tldw_cli/default_user/tldw_cli_app.log";
mkdirSync(outDir, { recursive: true });

async function settle(page, ms = 1500) {
  await page.evaluate(() => {
    document.body.classList.add("-first-byte");
    for (const el of document.querySelectorAll(".intro-dialog,.closed-dialog,.shade")) {
      el.style.pointerEvents = "none";
    }
  });
  await page.waitForTimeout(ms);
}

async function shot(page, name) {
  const path = resolve(outDir, `${name}.png`);
  await settle(page, 500);
  await page.screenshot({ path, fullPage: true });
  console.log(path);
}

function readLogTail(fromOffset = 0) {
  if (!existsSync(appLogPath)) {
    return "";
  }
  return readFileSync(appLogPath, "utf8").slice(fromOffset);
}

async function waitForLogAfter(offset, pattern, timeout = 45000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (pattern.test(readLogTail(offset))) {
      return;
    }
    await new Promise((resolveTimeout) => setTimeout(resolveTimeout, 250));
  }
  throw new Error(`Timed out waiting for app log pattern: ${pattern}`);
}

const browser = await chromium.launch({
  headless: true,
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
});
const page = await browser.newPage({
  viewport: { width: 2050, height: 1240 },
  deviceScaleFactor: 1,
});
await page.route("https://fonts.googleapis.com/**", (route) =>
  route.fulfill({ status: 200, contentType: "text/css", body: "" })
);

await page.goto(`http://127.0.0.1:${port}`, { waitUntil: "domcontentloaded", timeout: 30000 });
await page.waitForSelector("body", { state: "attached", timeout: 30000 });
await settle(page, 7000);
await shot(page, "01-idle-console");

await page.locator(".xterm-helper-textarea").focus();
await page.mouse.click(305, 1170);
await page.keyboard.type("Write one short sentence confirming the local llama.cpp Console path works.", { delay: 5 });
await shot(page, "02-typed-composer");

const logOffsetBeforeSend = existsSync(appLogPath) ? readFileSync(appLogPath, "utf8").length : 0;
await page.keyboard.press("Enter");
await page.waitForTimeout(1200);
await shot(page, "03-streaming-or-started");

await waitForLogAfter(logOffsetBeforeSend, /Updated message ID .*content = \?/);
await page.waitForTimeout(1500);
await shot(page, "04-completed-response");

await page.mouse.click(720, 350);
await page.waitForTimeout(1000);
await shot(page, "05-selected-message-actions");

await browser.close();
