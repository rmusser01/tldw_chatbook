import { createRequire } from "module";

const require = createRequire(import.meta.url);
const { chromium } = require("/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");

const browser = await chromium.launch({
  headless: true,
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
});
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
await page.route("https://fonts.googleapis.com/**", (route) =>
  route.fulfill({ status: 200, contentType: "text/css", body: "" })
);
try {
  await page.goto("http://127.0.0.1:8877", { waitUntil: "domcontentloaded", timeout: 10000 });
} catch (error) {
  console.log(`goto error: ${error}`);
}
await page.waitForTimeout(3000);
console.log(`url=${page.url()}`);
console.log(`frames=${page.frames().length}`);
console.log(`content-prefix=${(await page.content()).slice(0, 500)}`);
console.log(`body=${await page.evaluate(() => Boolean(document.body)).catch((error) => `eval-error:${error}`)}`);
await browser.close();
