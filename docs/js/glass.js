/* Glass micro-interactions — scroll reveal + pointer-reactive sheen + mesh hero.
   Works with Material's navigation.instant (re-init on each page load). */
(function () {
  // Inject the animated gradient-mesh blobs into any .sd-hero (no markup change)
  function mesh() {
    document.querySelectorAll(".sd-hero").forEach(function (hero) {
      if (hero.querySelector(".sd-mesh")) return;
      var m = document.createElement("div");
      m.className = "sd-mesh";
      m.setAttribute("aria-hidden", "true");
      for (var i = 1; i <= 4; i++) {
        var b = document.createElement("span");
        b.className = "sd-blob sd-blob--" + i;
        m.appendChild(b);
      }
      hero.insertBefore(m, hero.firstChild);
    });
  }

  function reveal() {
    var els = document.querySelectorAll(
      ".md-typeset .grid.cards > ul > li, .md-typeset .grid.cards > ol > li, " +
      ".md-typeset .grid > ul > li, " +
      ".md-typeset > table, .md-typeset .result, .md-typeset .admonition, " +
      ".md-typeset details, .md-typeset h2"
    );
    if (!("IntersectionObserver" in window)) {
      els.forEach(function (e) { e.classList.add("sd-in"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          en.target.classList.add("sd-in");
          io.unobserve(en.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.06 });
    els.forEach(function (e) { e.classList.add("sd-reveal"); io.observe(e); });
  }

  // Pointer-reactive sheen on glass cards (CSS reads --mx/--my)
  function sheen() {
    document.querySelectorAll(".md-typeset .grid.cards > ul > li," +
      ".md-typeset .grid.cards > ol > li," +
      ".md-typeset .grid > ul > li").forEach(function (card) {
      card.addEventListener("pointermove", function (ev) {
        var r = card.getBoundingClientRect();
        card.style.setProperty("--mx", ((ev.clientX - r.left) / r.width * 100) + "%");
        card.style.setProperty("--my", ((ev.clientY - r.top) / r.height * 100) + "%");
      });
    });
  }

  function init() { mesh(); reveal(); sheen(); }

  if (window.document$) {        // Material instant-nav: re-run per page
    window.document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
