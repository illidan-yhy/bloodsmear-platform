// Run the shipped htmx bundle and application code in a real browser against the test API.
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const { crc32 } = require('node:zlib');
const path = require('node:path');
const { chromium } = require('playwright');

const url = process.env.BLOODSMEAR_TEST_URL;
const png = Buffer.from(process.env.BLOODSMEAR_TEST_PNG_BASE64, 'base64');
const gray16 = Buffer.from(process.env.BLOODSMEAR_TEST_GRAY16_BASE64, 'base64');
const single = 'form[hx-post="/ui/infer"]';
const batch = 'form[hx-post="/ui/jobs"]';
let browser;

before(async () => {
  browser = await chromium.launch({
    headless: true,
    ...(process.env.BLOODSMEAR_BROWSER_CHANNEL ? { channel: process.env.BLOODSMEAR_BROWSER_CHANNEL } : {}),
  });
});
after(async () => { if (browser) await browser.close(); });

async function pageFor(t) {
  const page = await browser.newPage();
  page.setDefaultTimeout(4000);
  t.after(() => page.close());
  await page.goto(url);
  return page;
}

async function choose(page, filename, buffer = png, form = single) {
  // CDP transfer of the 26 MiB fixture is setup work, not the UI/server response deadline.
  await page.locator(`${form} input[type=file]`).setInputFiles(
    { name: filename, mimeType: 'image/png', buffer },
    { timeout: buffer.length > 1024 * 1024 ? 15000 : 4000 },
  );
}

async function submit(page, form = single) {
  await page.locator(`${form} button[type=submit]`).click();
}

async function errorText(page, target = '#result') {
  await page.locator(`${target} [role=alert]`).waitFor();
  return page.locator(target).innerText();
}

test('successful result identifies the actual uploaded Chinese filename', async t => {
  const page = await pageFor(t);
  await choose(page, '血涂片 样本1.png');
  await submit(page);
  await page.locator('#result img.annotated').waitFor();
  assert.match(await page.locator('#result').innerText(), /血涂片 样本1\.png/);
});

test('success then corrupt upload clears previous result before sending and recovers', async t => {
  const page = await pageFor(t);
  await choose(page, '血涂片 样本1.png');
  await submit(page);
  await page.locator('#result img.annotated').waitFor();
  let arrived;
  let release;
  const requested = new Promise(resolve => { arrived = resolve; });
  const resume = new Promise(resolve => { release = resolve; });
  await page.route('**/ui/infer', async route => { arrived(); await resume; await route.continue(); });
  await choose(page, 'corrupt.png', Buffer.from('not an image'));
  await submit(page);
  await requested;
  try {
    assert.equal(await page.locator('#result img').count(), 0);
    assert.equal(await page.locator('#result a[download]').count(), 0);
    assert.match(await page.locator('#result').innerText(), /处理中/);
    assert.equal(await page.locator(`${single} button[type=submit]`).isDisabled(), true);
  } finally { release(); }
  assert.ok(!(await errorText(page)).includes('INVALID_IMAGE'));
  assert.match(await page.locator('#result').innerText(), /损坏|无法读取/);
  assert.equal(await page.locator('#result img').count(), 0);
  assert.equal(await page.locator('#result a[download]').count(), 0);
  const alertBox = await page.locator('#result [role=alert]').boundingBox();
  assert.ok(alertBox.y >= 0 && alertBox.y + alertBox.height <= page.viewportSize().height,
    'The new error must be visible, not hidden below the batch form');
  await page.screenshot({ path: path.join(process.env.BLOODSMEAR_TEST_ARTIFACTS, 'corrupt-image-feedback.png') });
  await page.unroute('**/ui/infer');
  await choose(page, '恢复正常.png');
  await submit(page);
  await page.locator('#result img.annotated').waitFor();
  assert.equal(await page.locator('#result [role=alert]').count(), 0);
});

test('grouped batch without sample ID shows an error instead of its previous task', async t => {
  const page = await pageFor(t);
  await choose(page, 'sample.png', png, batch);
  await submit(page, batch);
  await page.locator('#batch-result .batch-status').waitFor();
  await page.locator(`${batch} input[value=grouped]`).check();
  await submit(page, batch);
  const text = await errorText(page, '#batch-result');
  assert.ok(!text.includes('GROUP_SAMPLE_ID_REQUIRED'));
  assert.match(text, /样本.*ID|样本编号/);
  assert.equal(await page.locator('#batch-result .batch-status').count(), 0);
});

for (const scenario of [
  { name: 'unsupported BMP', filename: 'cell.bmp', buffer: png, message: /格式|JPG/ },
  { name: '16-bit grayscale TIFF', filename: '相机16位.tif', buffer: gray16, message: /不支持.*16.*灰度/ },
  { name: 'file over 25 MiB', filename: 'too-big.png', buffer: Buffer.alloc(25 * 1024 * 1024 + 1), message: /25.*MiB|25.*MB/ },
  { name: '14000x14000 pixel image with a tiny payload', filename: 'huge.png', buffer: (() => {
    const data = Buffer.from(png);
    data.writeUInt32BE(14000, 16);
    data.writeUInt32BE(14000, 20);
    data.writeUInt32BE(crc32(data.subarray(12, 29)), 29);
    return data;
  })(), message: /像素|尺寸/ },
]) {
  test(`${scenario.name} shows an actionable image error`, async t => {
    const page = await pageFor(t);
    await choose(page, scenario.filename, scenario.buffer);
    await submit(page);
    const text = await errorText(page);
    assert.ok(!text.includes('INVALID_IMAGE'));
    assert.match(text, scenario.message);
    assert.equal(await page.locator('#result img').count(), 0);
    assert.equal(await page.locator(`${single} button[type=submit]`).isDisabled(), false);
  });
}

for (const scenario of [
  { name: 'GPU unavailable', status: 503, body: JSON.stringify({ error: { code: 'GPU_UNAVAILABLE', message: 'private GPU path' } }), contentType: 'application/json', message: /GPU.*不可用|GPU.*无法/ },
  { name: 'bare internal server error', status: 500, body: 'Internal Server Error private traceback', contentType: 'text/plain', message: /服务器|服务端/ },
  { name: 'form validation error', status: 422, body: JSON.stringify({ detail: [{ msg: 'private validation detail' }] }), contentType: 'application/json', message: /填写|信息|格式/ },
  { name: 'request entity too large', status: 413, body: 'Request Entity Too Large', contentType: 'text/plain', message: /过大|大小|超过/ },
]) {
  test(`${scenario.name} is displayed without backend details`, async t => {
    const page = await pageFor(t);
    await page.route('**/ui/infer', route => route.fulfill({
      status: scenario.status, body: scenario.body, contentType: scenario.contentType,
    }));
    await choose(page, 'sample.png');
    await submit(page);
    const text = await errorText(page);
    assert.match(text, scenario.message);
    assert.ok(!text.includes('private'));
    assert.equal(await page.locator(`${single} button[type=submit]`).isDisabled(), false);
  });
}

test('network failure shows a connection error and allows another submission', async t => {
  const page = await pageFor(t);
  await page.route('**/ui/infer', route => route.abort('failed'));
  await choose(page, 'sample.png');
  await submit(page);
  assert.match(await errorText(page), /网络|连接/);
  assert.equal(await page.locator(`${single} button[type=submit]`).isDisabled(), false);
});

test('mixed batch upload identifies the rejected file and says no task was created', async t => {
  const page = await pageFor(t);
  await page.locator(`${batch} input[type=file]`).setInputFiles([
    { name: '正常 图片.png', mimeType: 'image/png', buffer: png },
    { name: '<错误>.bmp', mimeType: 'image/bmp', buffer: png },
  ]);
  await submit(page, batch);
  const text = await errorText(page, '#batch-result');
  assert.match(text, /未创建/);
  const rows = await page.locator('#batch-result table tbody tr').allTextContents();
  assert.equal(rows.length, 1);
  assert.match(rows[0], /2/);
  assert.ok(rows[0].includes('<错误>.bmp'));
  assert.match(rows[0], /格式/);
  assert.ok(!rows[0].includes('正常 图片.png'));
  assert.ok(!text.includes('INVALID_IMAGE'));
  assert.equal(await page.locator('#batch-result table img').count(), 0);
  await page.screenshot({ path: path.join(process.env.BLOODSMEAR_TEST_ARTIFACTS, 'mixed-batch-file-feedback.png') });
});
