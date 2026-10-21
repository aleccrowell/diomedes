// Check that every figure on the built site renders (#56).
//
//   node docs/test/render_figures.js <base URL> <build dir>
//
// Serve docs/build at <base URL> first (e.g. python3 -m http.server). Every built
// page that embeds a figure (diomedesFigure) is loaded in jsdom with its own
// scripts: Documenter's RequireJS, Vega from the CDN, docs/src/assets/figures.js.
// That is how the site failed before #53 (Vega registering as an anonymous AMD
// module), which a check that renders the specs directly can't see. The check
// passes when every figure container holds an SVG, every data download link
// (a.diomedes-data) resolves and every number computed from the data
// (span.diomedes-stat) is filled in; it exits 1 otherwise.
// jsdom lacks structuredClone and matchMedia, which browsers have: polyfilled.

const fs = require("fs");
const path = require("path");
const { JSDOM, VirtualConsole } = require("jsdom");

const [base, buildDir] = process.argv.slice(2);
if (!base || !buildDir) { console.error("usage: render_figures.js <base URL> <build dir>"); process.exit(2); }
const TIMEOUT_MS = 120000;

function pagesWithFigures(dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...pagesWithFigures(p));
    else if (e.name.endsWith(".html") && /diomedesFigure\("|class="diomedes-data"|class=(&quot;|")diomedes-stat/.test(fs.readFileSync(p, "utf8"))) out.push(p);
  }
  return out;
}

async function checkPage(file) {
  const rel = path.relative(buildDir, file).split(path.sep).join("/");
  const errors = [];
  const vc = new VirtualConsole();
  vc.on("jsdomError", e => errors.push(String(e.message || e).slice(0, 200)));
  const dom = await JSDOM.fromURL(base.replace(/\/?$/, "/") + rel, {
    runScripts: "dangerously", resources: "usable", pretendToBeVisual: true, virtualConsole: vc,
    beforeParse(w) {
      w.fetch = (u, o) => fetch(new URL(u, w.location.href), o);
      w.structuredClone = structuredClone;
      w.matchMedia = () => ({ matches: false, addListener() {}, removeListener() {}, addEventListener() {} });
    },
  });
  const doc = dom.window.document;
  const start = Date.now();
  while (doc.readyState !== "complete" && Date.now() - start < TIMEOUT_MS) await new Promise(r => setTimeout(r, 200));
  let figs = [];
  while (Date.now() - start < TIMEOUT_MS) {
    figs = [...doc.querySelectorAll(".diomedes-figure")];
    const stats = [...doc.querySelectorAll("span.diomedes-stat")];
    if (figs.every(f => f.querySelector("svg") || f.textContent.startsWith("Figure failed")) &&
        stats.every(e => e.classList.contains("diomedes-stat-filled") || e.textContent === "?")) break;
    await new Promise(r => setTimeout(r, 1000));
  }
  const links = [];
  for (const a of doc.querySelectorAll("a.diomedes-data")) {
    const href = a.getAttribute("href");
    let status = "unresolved";
    if (href && href !== "#") {
      // HEAD: only the status is needed. An unread GET body larger than undici's buffer
      // leaves its parser paused, and the server closing the connection then fails an
      // internal assertion (`assert(!this.paused)`), crashing the check
      try {
        const r = await fetch(new URL(href, dom.window.location.href), { method: "HEAD" });
        status = r.status;
        await r.body?.cancel();
      } catch (e) { status = String(e); }
    }
    links.push({ path: a.dataset.path, ok: status === 200, status });
  }
  const results = figs.map(f => ({ id: f.id, ok: !!f.querySelector("svg"),
                                   detail: f.querySelector("svg") ? "" : (f.textContent || "empty").slice(0, 160) }));
  // windows stay open until the end: a figure can re-render after the check (theme
  // changes), and its callbacks must not run against a closed window
  const statResults = [...doc.querySelectorAll("span.diomedes-stat")].map(e => ({
    what: `${e.dataset.stat}(${[e.dataset.x, e.dataset.y].filter(Boolean).join(", ")}) of ${e.dataset.path}`,
    ok: e.classList.contains("diomedes-stat-filled") && e.textContent !== "?", value: e.textContent }));
  // a span written in a Markdown paragraph, not a raw HTML block, is escaped into text
  if (doc.body.textContent.includes('<span class="diomedes-stat"'))
    statResults.push({ what: "markup: a diomedes-stat span was escaped (put it in a raw html block)", ok: false, value: "" });
  return { page: rel, results, links, stats: statResults, errors };
}

(async () => {
  const pages = pagesWithFigures(buildDir);
  if (!pages.length) { console.error("no pages with figures under " + buildDir); process.exit(1); }
  let failed = 0, total = 0;
  for (const p of pages) {
    const { page, results, links, stats, errors } = await checkPage(p);
    for (const t of stats) {
      total++;
      if (t.ok) console.log(`ok   ${page} stat ${t.what} = ${t.value}`);
      else { failed++; console.log(`FAIL ${page} stat ${t.what}: ${t.value}`); }
    }
    for (const l of links) {
      total++;
      if (l.ok) console.log(`ok   ${page} data ${l.path}`);
      else { failed++; console.log(`FAIL ${page} data ${l.path}: ${l.status}`); }
    }
    for (const r of results) {
      total++;
      if (r.ok) console.log(`ok   ${page} ${r.id}`);
      else { failed++; console.log(`FAIL ${page} ${r.id}: ${r.detail}`); }
    }
    if (results.some(r => !r.ok) && errors.length) console.log("     page errors:", [...new Set(errors)].slice(0, 5));
  }
  console.log(`${total - failed}/${total} checks passed (figures rendered, data links resolved, numbers filled) on ${pages.length} pages`);
  process.exit(failed ? 1 : 0);
})();
