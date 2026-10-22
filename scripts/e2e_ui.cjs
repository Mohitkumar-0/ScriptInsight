/* Browser end-to-end checks for the ScriptInsight UI.
 *
 * Requires Node 18+ and playwright-core (not a project dependency), e.g.:
 *   mkdir -p /tmp/si-e2e && cd /tmp/si-e2e && npm i playwright-core@1.63.0 && npx playwright-core install webkit
 *   (older Playwright releases could not download WebKit for macOS 27)
 *   NODE_PATH=/tmp/si-e2e/node_modules node scripts/e2e_ui.cjs http://127.0.0.1:5001 chrome
 *   NODE_PATH=/tmp/si-e2e/node_modules node scripts/e2e_ui.cjs http://127.0.0.1:5001 webkit
 *   # with a server started with SCRIPTINSIGHT_ENABLE_LEGACY_MODEL=false:
 *   NODE_PATH=... node scripts/e2e_ui.cjs http://127.0.0.1:5002 chrome --no-models
 *
 * "chrome" uses the locally installed Google Chrome; "webkit" uses Playwright's WebKit (Safari engine).
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { chromium, webkit } = require('playwright-core');

const BASE = process.argv[2] || 'http://127.0.0.1:5001';
const ENGINE = process.argv[3] || 'chrome';
const NO_MODELS = process.argv.includes('--no-models');
const ROOT = path.resolve(__dirname, '..');
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'si-e2e-'));

let failures = 0;
const ok = (name, cond, extra = '') => {
  if (!cond) failures++;
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name} ${extra}`);
};

function writeFixtures() {
  fs.writeFileSync(path.join(TMP, 'notes.txt'), 'not an image');
  fs.writeFileSync(path.join(TMP, 'corrupt.png'), 'definitely not a png');
  fs.writeFileSync(path.join(TMP, 'big.jpg'), Buffer.alloc(17 * 1024 * 1024));
}

(async () => {
  writeFixtures();
  const browser = ENGINE === 'webkit' ? await webkit.launch() : await chromium.launch({ channel: 'chrome' });
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage();
  const consoleErrors = [];
  let predictRequests = 0;
  let choosers = 0;
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));
  page.on('request', (r) => { if (r.url().endsWith('/api/predict')) predictRequests++; });
  page.on('filechooser', () => choosers++);
  await page.addInitScript(() => {
    window.__sent = [];
    const original = window.fetch;
    window.fetch = (url, opts) => {
      const file = opts && opts.body instanceof FormData ? opts.body.get('file') : null;
      if (file) window.__sent.push({ name: file.name, size: file.size, keys: [...opts.body.keys()] });
      return original(url, opts);
    };
  });

  const msg = () => page.textContent('#fileMessage');
  const disabled = () => page.$eval('#analyzeBtn', (b) => b.disabled);
  const label = () => page.textContent('#selectedFileLabel');
  const waitMsg = (re) => page.waitForFunction((src) => new RegExp(src).test(document.querySelector('#fileMessage').textContent), re.source, { timeout: 60000 });

  await page.goto(`${BASE}/#analyze`);
  ok('1. initial: Analyze disabled with guidance', await disabled() && /Choose or drop/.test(await msg()));
  await page.$eval('#uploadForm', (f) => f.requestSubmit());
  ok('1b. forced submit without image shows validation', /choose an image file first/i.test(await msg()) && predictRequests === 0);

  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.click('#uploadBox')]);
  await page.waitForTimeout(300);
  ok('2a. one click opens the picker exactly once', choosers === 1, `(count=${choosers})`);
  await chooser.setFiles(path.join(ROOT, '1.1.jpg'));
  await waitMsg(/ready for analysis/);
  const meta = await page.textContent('#fileMeta');
  ok('2b. filename, size, dimensions, preview, enabled', await label() === '1.1.jpg' && /900×1600/.test(meta)
    && !(await disabled()) && await page.isVisible('#selectedPreview') && await page.isVisible('#replaceBtn'), `(${meta})`);

  await page.setInputFiles('#fileInput', []);
  ok('3. cancelled picker keeps the selection', await label() === '1.1.jpg' && !(await disabled()));

  if (NO_MODELS) {
    await page.click('#analyzeBtn');
    await waitMsg(/No prediction model/);
    ok('M1. no models: clear 503 message, button re-enabled', !(await disabled()), `→ "${await msg()}"`);
    await page.click('#tab-dashboard');
    await page.waitForSelector('.model-list li');
    const chips = await page.$$eval('.model-list .chip', (els) => els.map((e) => e.textContent));
    ok('M2. dashboard marks every model unavailable', chips.length === 4 && chips.every((c) => c !== 'Loaded'), JSON.stringify(chips));
  } else {
    await page.click('#analyzeBtn');
    const busy = await page.$eval('#analyzeBtn', (b) => [b.disabled, b.textContent]);
    await page.click('#analyzeBtn', { force: true, noWaitAfter: true }).catch(() => {});
    await waitMsg(/Analysis complete/);
    const sent = await page.evaluate(() => window.__sent);
    ok('4a. loading state while analysing', busy[0] && /Analyzing/.test(busy[1]));
    ok('4b. FormData field "file" carries the selected image', sent.length === 1 && sent[0].name === '1.1.jpg'
      && sent[0].size === fs.statSync(path.join(ROOT, '1.1.jpg')).size && sent[0].keys.join() === 'file', JSON.stringify(sent));
    ok('4c. duplicate click sent one request', predictRequests === 1, `(requests=${predictRequests})`);
    const trait = await page.textContent('#predictedTrait');
    const bars = await page.$$eval('#probabilityGrid [role=progressbar]', (els) => els.map((e) => e.getAttribute('aria-label')));
    ok('4d. result shows legacy model, five accessible bars in OCEAN order', /Legacy ResNet50/.test(await page.textContent('#resultModel'))
      && bars.join() === 'Openness,Conscientiousness,Extraversion,Agreeableness,Neuroticism', `→ ${trait}`);
    const rows = await page.$$eval('#comparisonTable tbody tr', (trs) => trs.map((t) => t.textContent.replace(/\s+/g, ' ').trim().slice(0, 60)));
    ok('4e. comparison lists 4 models, 3 marked unavailable', rows.length === 4 && rows.filter((r) => /Unavailable/.test(r)).length === 3);
    ok('4f. limitation warning shown with result', /not independent trait levels/.test(await page.textContent('#resultNotes')));

    const [download] = await Promise.all([page.waitForEvent('download'), page.click('#exportBtn')]);
    const reportPath = path.join(TMP, 'report.html');
    await download.saveAs(reportPath);
    const report = fs.readFileSync(reportPath, 'utf8');
    ok('5. exported report has disclaimer, model and image info', /Experimental/.test(report) && /Legacy ResNet50/.test(report) && /1\.1\.jpg/.test(report));
  }

  await page.setInputFiles('#fileInput', path.join(TMP, 'notes.txt'));
  ok('6a. unsupported type rejected, previous image kept', /Unsupported image type/.test(await msg()) && await label() === '1.1.jpg');
  await page.setInputFiles('#fileInput', path.join(TMP, 'big.jpg'));
  ok('6b. oversized file rejected in the browser', /too large/.test(await msg()));
  await page.setInputFiles('#fileInput', { name: 'tiny.png', mimeType: 'image/png',
    buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAAAAACPAi4CAAAAJ0lEQVR42u3OMQ0AAAgDsM2/aDDBQUKqoE3yXAcBAQEBAQEBAQEBAQGBdwHbAABAqsbgBAAAAABJRU5ErkJggg==', 'base64') });
  await waitMsg(/too small/);
  ok('6c. image below 128x128 rejected in the browser', await label() === '1.1.jpg', `→ "${await msg()}"`);

  // Drag and drop a second image.
  const dropped = fs.readFileSync(path.join(ROOT, '3.3.jpg')).toString('base64');
  await page.evaluate(async (b64) => {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const dt = new DataTransfer();
    dt.items.add(new File([bytes], '3.3.jpg', { type: 'image/jpeg' }));
    const box = document.getElementById('uploadBox');
    box.dispatchEvent(new DragEvent('dragover', { dataTransfer: dt, bubbles: true, cancelable: true }));
    box.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }));
  }, dropped);
  await waitMsg(/ready for analysis/);
  ok('7a. drag-and-drop replaces the image', await label() === '3.3.jpg');

  if (!NO_MODELS) {
    await page.click('#analyzeBtn');
    await waitMsg(/Analysis complete/);
    const last = await page.evaluate(() => window.__sent.at(-1));
    ok('7b. second analysis sends the replacement image', last.name === '3.3.jpg', `→ ${await page.textContent('#predictedTrait')}`);

    await page.setInputFiles('#fileInput', path.join(TMP, 'corrupt.png'));
    await waitMsg(/ready for analysis/);
    await page.click('#analyzeBtn');
    await waitMsg(/corrupted/);
    ok('8a. corrupted image: server message shown', true, `→ "${await msg()}"`);

    await page.setInputFiles('#fileInput', path.join(ROOT, '1.2.jpg'));
    await waitMsg(/ready for analysis/);
    await page.route('**/api/predict', (r) => r.fulfill({ status: 500, contentType: 'application/json', body: '{"success":false,"error":"The image could not be analysed because of a server error.","code":"server_error"}' }));
    await page.click('#analyzeBtn');
    await waitMsg(/server error/);
    ok('8b. backend JSON error shown', true);
    await page.unroute('**/api/predict');
    await page.route('**/api/predict', (r) => r.fulfill({ status: 502, contentType: 'text/html', body: '<html>Bad gateway</html>' }));
    await page.click('#analyzeBtn');
    await waitMsg(/HTTP 502/);
    ok('8c. non-JSON error shown', true);
    await page.unroute('**/api/predict');
    await page.route('**/api/predict', (r) => r.abort('connectionrefused'));
    await page.click('#analyzeBtn');
    await waitMsg(/reach the server/);
    ok('8d. network failure shown, button re-enabled', !(await disabled()));
    await page.unroute('**/api/predict');

    await page.click('#tab-history');
    const items = await page.$$eval('.history-items li', (els) => els.length);
    ok('9a. history lists the two successful analyses', items === 2, `(items=${items})`);
    await page.click('#clearHistoryBtn');
    ok('9b. clear history empties the list', /No analyses/.test(await page.textContent('#historyList')));

    await page.click('#tab-dashboard');
    await page.waitForSelector('.model-list li');
    const dash = await page.textContent('#dashboardContent');
    ok('10a. dashboard: model states and recorded legacy metrics with provenance',
      /Not trained/.test(dash) && /30\.3%/.test(dash) && /Recorded in notebook/.test(dash) && /Not measurable yet/.test(dash));
    await page.click('#tab-evaluation');
    await page.waitForSelector('#evaluationContent .line-chart');
    const evalText = await page.textContent('#evaluationContent');
    ok('10b. evaluation: empty state for new models + recorded legacy charts',
      /No evaluation results yet/.test(evalText) && (await page.$$('#evaluationContent .line-chart')).length === 2);
  }

  await page.click('#tab-analyze');
  await page.focus('#tab-analyze');
  await page.keyboard.press('ArrowRight');
  ok('11a. arrow keys move between tabs', await page.evaluate(() => document.activeElement.id) === 'tab-dashboard');
  await page.click('#tab-analyze');
  await page.focus('#uploadBox');
  const before = choosers;
  await Promise.all([page.waitForEvent('filechooser', { timeout: 5000 }).catch(() => null), page.keyboard.press('Enter')]);
  ok('11b. Enter on the focused upload box opens the picker', choosers === before + 1);

  await page.setViewportSize({ width: 375, height: 800 });
  for (const tab of ['analyze', 'dashboard', 'evaluation', 'history']) {
    await page.click(`#tab-${tab}`);
    await page.waitForTimeout(200);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    ok(`12. 375px wide, no horizontal scroll on ${tab}`, overflow <= 0, `(overflow=${overflow}px)`);
  }
  await page.click('#tab-analyze');
  await page.screenshot({ path: path.join(TMP, `mobile-${ENGINE}.png`), fullPage: false });

  const appErrors = consoleErrors.filter((e) => !/Failed to load resource|ERR_CONNECTION_REFUSED|Could not connect|status of (4|5)\d\d/.test(e));
  ok('13. no console errors from application code', appErrors.length === 0, JSON.stringify(appErrors));
  await browser.close();
  console.log(`\n${ENGINE}: ${failures ? `${failures} FAILED` : 'all checks passed'} (screenshots in ${TMP})`);
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
