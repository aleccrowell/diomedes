// Interactive figures for the Methodology and Findings pages (#45).
//
// A page embeds a Vega-Lite spec with
//   <div id="fig-x" class="diomedes-figure"></div>
//   <script>diomedesFigure("fig-x", { ...spec... });</script>
// Data URLs in the spec are relative to docs/src (e.g. "assets/data/race/drivers.csv")
// and are resolved against Documenter's base URL, so they work whatever the page's
// depth. Charts follow the site theme: they re-render when the theme changes.

(function () {
  const figures = [];

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
          ? (typeof documenterBaseURL === "string" ? documenterBaseURL : ".") + "/" + v
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
    vegaEmbed("#" + fig.id, spec, {
      theme: dark ? "dark" : undefined,
      actions: { export: true, source: false, compiled: false, editor: true },
      renderer: "svg",
    }).catch((err) => {
      document.getElementById(fig.id).textContent = "Figure failed to load: " + err;
    });
  }

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
