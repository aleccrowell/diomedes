// Interactive figures for the Methodology and Findings pages (#45).
//
// A page embeds a Vega-Lite spec with
//   <div id="fig-x" class="diomedes-figure"></div>
//   <script>diomedesFigure("fig-x", { ...spec... });</script>
// Data URLs in the spec are relative to docs/src (e.g. "assets/data/race/drivers.csv")
// and are resolved against Documenter's base URL, so they work whatever the page's
// depth. Charts follow the site theme: they re-render when the theme changes.
// Specs use width "container", so charts fit the page column (phones included).
//
// Download links: <a class="diomedes-data" data-path="assets/data/race/drivers.csv">
// gets its href resolved the same way.
//
// Documenter pages load RequireJS before any other script. The Vega builds are
// UMD modules: with RequireJS present they register as anonymous AMD modules
// instead of setting globals (and RequireJS rejects anonymous modules loaded by
// plain <script> tags). So Vega is loaded through RequireJS here, never as
// page assets; without RequireJS it falls back to script tags.

(function () {
  const figures = [];
  const CDN = "https://cdn.jsdelivr.net/npm/";
  const LIBS = { "vega": "vega@5.30.0/build/vega.min", "vega-lite": "vega-lite@5.21.0/build/vega-lite.min",
                 "vega-embed": "vega-embed@6.26.0/build/vega-embed.min" };
  let embedPromise = null;

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src; s.onload = resolve; s.onerror = () => reject(new Error("could not load " + src));
      document.head.appendChild(s);
    });
  }

  function getVegaEmbed() {
    if (embedPromise) return embedPromise;
    embedPromise = new Promise((resolve, reject) => {
      if (typeof window.requirejs === "function") {
        const paths = {};
        for (const [k, v] of Object.entries(LIBS)) paths[k] = CDN + v;
        window.requirejs.config({ paths });
        window.requirejs(["vega-embed"], resolve, reject);
      } else {
        loadScript(CDN + LIBS["vega"] + ".js")
          .then(() => loadScript(CDN + LIBS["vega-lite"] + ".js"))
          .then(() => loadScript(CDN + LIBS["vega-embed"] + ".js"))
          .then(() => resolve(window.vegaEmbed), reject);
      }
    });
    return embedPromise;
  }

  function isDark() {
    // Documenter marks the theme with a class on <html>; read the actual background
    // instead of matching theme names
    const bg = getComputedStyle(document.body).backgroundColor.match(/\d+/g);
    if (!bg) return false;
    const [r, g, b] = bg.map(Number);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 128;
  }

  function resolveUrls(node) {
    if (Array.isArray(node)) return node.map(resolveUrls);
    if (node && typeof node === "object") {
      const out = {};
      for (const [k, v] of Object.entries(node)) {
        out[k] = k === "url" && typeof v === "string" && !/^[a-z]+:/i.test(v)
          ? baseURL() + "/" + v
          : resolveUrls(v);
      }
      return out;
    }
    return node;
  }

  function render(fig) {
    const dark = isDark();
    const spec = Object.assign({ $schema: "https://vega.github.io/schema/vega-lite/v5.json" }, resolveUrls(fig.spec));
    spec.config = Object.assign({ background: null, view: { stroke: null } }, spec.config || {});
    getVegaEmbed()
      .then((vegaEmbed) => vegaEmbed("#" + fig.id, spec, {
        theme: dark ? "dark" : undefined,
        actions: { export: true, source: false, compiled: false, editor: true },
        renderer: "svg",
      }))
      .catch((err) => {
        const el = document.getElementById(fig.id);
        if (el) el.textContent = "Figure failed to load: " + err;
      });
  }

  // figure containers fill the content column, so "container" widths fit it
  const style = document.createElement("style");
  style.textContent = ".diomedes-figure { width: 100%; margin: 1em 0; } .diomedes-figure .vega-embed { width: 100%; }";
  document.head.appendChild(style);

  function baseURL() { return typeof documenterBaseURL === "string" ? documenterBaseURL : "."; }

  function resolveDataLinks() {
    for (const a of document.querySelectorAll("a.diomedes-data[data-path]")) {
      a.href = baseURL() + "/" + a.dataset.path;
      a.setAttribute("download", a.dataset.path.split("/").pop());
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", resolveDataLinks);
  else resolveDataLinks();

  window.diomedesFigure = function (id, spec) {
    const fig = { id, spec };
    figures.push(fig);
    if (document.readyState === "complete") render(fig);
    else window.addEventListener("load", () => render(fig));
  };

  // re-render on theme changes
  new MutationObserver(() => figures.forEach(render))
    .observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
})();
