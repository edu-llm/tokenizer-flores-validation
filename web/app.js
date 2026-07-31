(function () {
  const data = window.METRICS_DATA;
  if (!data) {
    document.body.innerHTML =
      "<p>Missing data.js — run <code>python scripts/export_web_data.py</code>.</p>";
    return;
  }

  const caption = document.getElementById("caption");
  const tabs = document.querySelectorAll(".tab");
  const tabProduction = document.getElementById("tab-production");
  const tabExperiment = document.getElementById("tab-experiment");

  const prodSelect = document.getElementById("metric-select");
  const prodExplanation = document.getElementById("explanation");
  const prodHeatmap = document.getElementById("heatmap");
  const prodHeatmapTitle = document.getElementById("heatmap-title");

  const expSectionsContainer = document.getElementById("experiment-sections");
  const experiments =
    data.experiments || (data.experiment ? [data.experiment] : []);

  caption.textContent =
    data.source +
    " · " +
    data.n_sentences +
    " sentences · " +
    data.languages.length +
    " languages · " +
    data.tokenizers.length +
    " tokenizers";

  function fillMetricSelect(select) {
    select.innerHTML = "";
    for (const m of data.metrics) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      select.appendChild(opt);
    }
    select.value = data.metrics[0].id;
  }

  fillMetricSelect(prodSelect);

  const prodLookup = new Map();
  for (const row of data.rows) {
    prodLookup.set(row.tokenizer_id + "|" + row.language, row);
  }

  function getProdValue(metricId, tokId, langCode) {
    const row = prodLookup.get(tokId + "|" + langCode);
    if (!row) return null;
    const v = row[metricId];
    return v === null || v === undefined ? null : v;
  }

  function makeGetValue(rows) {
    const lookup = new Map();
    for (const row of rows) {
      lookup.set(row.tokenizer_id + "|" + row.language, row);
    }
    return function (metricId, armId, langCode) {
      const row = lookup.get(armId + "|" + langCode);
      if (!row) return null;
      const v = row[metricId];
      return v === null || v === undefined ? null : v;
    };
  }

  function formatValue(v) {
    if (v === null || Number.isNaN(v)) return "—";
    return Number(v).toFixed(3);
  }

  function severity(metric, values) {
    const nums = values.filter((v) => v !== null && !Number.isNaN(v));
    if (!nums.length) return values.map(() => null);
    const lo = Math.min(...nums);
    const hi = Math.max(...nums);
    const span = hi - lo || 1;
    return values.map((v) => {
      if (v === null || Number.isNaN(v)) return null;
      const norm = (v - lo) / span;
      return metric.higher_is_worse ? norm : 1 - norm;
    });
  }

  function heatColor(sev) {
    if (sev === null) return "#e7e5e4";
    const t = Math.max(0, Math.min(1, sev));
    const r = Math.round(255 - t * (255 - 185));
    const g = Math.round(247 - t * (247 - 28));
    const b = Math.round(237 - t * (237 - 28));
    return `rgb(${r},${g},${b})`;
  }

  function meanNumeric(values) {
    const nums = values.filter((v) => v !== null && !Number.isNaN(v));
    if (!nums.length) return null;
    return nums.reduce((a, b) => a + b, 0) / nums.length;
  }

  function badnessScore(metric, values) {
    const m = meanNumeric(values);
    if (m === null) return Number.NEGATIVE_INFINITY;
    return metric.higher_is_worse ? m : -m;
  }

  function applicableLanguages(metric, languages, columns, getValue) {
    const omit = new Set(metric.omit_languages || []);
    return languages.filter((lang) => {
      if (omit.has(lang.code)) return false;
      return columns.some((col) => getValue(metric.id, col.id, lang.code) !== null);
    });
  }

  function sortedAxes(metric, languages, columns, getValue) {
    const langsAvail = applicableLanguages(metric, languages, columns, getValue);
    const cols = [...columns].sort((a, b) => {
      const av = langsAvail.map((l) => getValue(metric.id, a.id, l.code));
      const bv = langsAvail.map((l) => getValue(metric.id, b.id, l.code));
      return badnessScore(metric, bv) - badnessScore(metric, av);
    });
    const langs = [...langsAvail].sort((a, b) => {
      const av = cols.map((c) => getValue(metric.id, c.id, a.code));
      const bv = cols.map((c) => getValue(metric.id, c.id, b.code));
      return badnessScore(metric, bv) - badnessScore(metric, av);
    });
    return { cols, langs };
  }

  function renderHeatmap(container, titleEl, metric, languages, columns, getValue) {
    titleEl.textContent = metric.label + " — heatmap (worst → best; redder = worse)";
    const { cols, langs } = sortedAxes(metric, languages, columns, getValue);

    const flat = [];
    for (const lang of langs) {
      for (const col of cols) {
        flat.push(getValue(metric.id, col.id, lang.code));
      }
    }
    const sevs = severity(metric, flat);

    container.style.gridTemplateColumns = `minmax(6.5rem, 9rem) repeat(${cols.length}, minmax(3.4rem, 1fr))`;
    container.innerHTML = "";

    const corner = document.createElement("div");
    corner.className = "cell corner";
    container.appendChild(corner);
    for (const col of cols) {
      const h = document.createElement("div");
      h.className = "cell col-head";
      h.textContent = col.label;
      container.appendChild(h);
    }

    let idx = 0;
    for (const lang of langs) {
      const rh = document.createElement("div");
      rh.className = "cell row-head";
      rh.textContent = lang.name;
      rh.title = lang.code;
      container.appendChild(rh);
      for (const col of cols) {
        const v = flat[idx];
        const cell = document.createElement("div");
        cell.className = "cell";
        cell.style.background = heatColor(sevs[idx]);
        cell.textContent = formatValue(v);
        cell.title = `${lang.name} · ${col.label}: ${formatValue(v)}`;
        container.appendChild(cell);
        idx += 1;
      }
    }
  }

  function renderProduction() {
    const metric = data.metrics.find((m) => m.id === prodSelect.value);
    prodExplanation.textContent = metric.explanation;
    renderHeatmap(
      prodHeatmap,
      prodHeatmapTitle,
      metric,
      data.languages,
      data.tokenizers,
      getProdValue
    );
  }

  function buildExperimentSections() {
    if (!expSectionsContainer) return;
    expSectionsContainer.innerHTML = "";

    for (const exp of experiments) {
      const section = document.createElement("section");
      section.className = "experiment-section";

      const title = document.createElement("h2");
      title.className = "experiment-title";
      title.textContent = exp.title || exp.source;
      section.appendChild(title);

      if (exp.source && exp.title) {
        const sub = document.createElement("p");
        sub.className = "caption";
        sub.textContent = exp.source;
        section.appendChild(sub);
      }

      const explanation = document.createElement("p");
      explanation.className = "experiment-explanation";
      explanation.textContent = exp.explanation;
      section.appendChild(explanation);

      if (exp.plot) {
        const img = document.createElement("img");
        img.className = "experiment-plot";
        img.src = exp.plot;
        img.alt = (exp.title || "experiment") + " — gap vs vocab size";
        section.appendChild(img);
      }

      const controls = document.createElement("div");
      controls.className = "controls";
      const label = document.createElement("label");
      label.textContent = "Score";
      const select = document.createElement("select");
      fillMetricSelect(select);
      const metricExplanation = document.createElement("aside");
      metricExplanation.className = "explanation";
      metricExplanation.setAttribute("aria-live", "polite");
      controls.appendChild(label);
      controls.appendChild(select);
      controls.appendChild(metricExplanation);
      section.appendChild(controls);

      const panel = document.createElement("section");
      panel.className = "panel heatmap-panel";
      const heatTitle = document.createElement("h3");
      const heatmapDiv = document.createElement("div");
      heatmapDiv.className = "heatmap";
      const legend = document.createElement("p");
      legend.className = "legend-note";
      legend.textContent =
        "Color scales by severity within this metric. Redder = worse for the selected score.";
      panel.appendChild(heatTitle);
      panel.appendChild(heatmapDiv);
      panel.appendChild(legend);
      section.appendChild(panel);

      expSectionsContainer.appendChild(section);

      const getValue = makeGetValue(exp.rows);
      function renderThisSection() {
        const metric = data.metrics.find((m) => m.id === select.value);
        metricExplanation.textContent = metric.explanation;
        renderHeatmap(
          heatmapDiv,
          heatTitle,
          metric,
          exp.languages,
          exp.arms,
          getValue
        );
      }
      select.addEventListener("change", renderThisSection);
      renderThisSection();
    }
  }

  // --- Zipf deviation section -------------------------------------------
  const zipf = data.zipf;
  const zipfBody = document.getElementById("zipf-body");
  const zipfMissing = document.getElementById("zipf-missing");
  const zipfViewSelect = document.getElementById("zipf-view-select");
  const zipfMetricSelect = document.getElementById("zipf-metric-select");
  const zipfViewExplanation = document.getElementById("zipf-view-explanation");
  const zipfMetricExplanation = document.getElementById("zipf-metric-explanation");
  const zipfHeatmap = document.getElementById("zipf-heatmap");
  const zipfHeatmapTitle = document.getElementById("zipf-heatmap-title");

  function pct(v, digits) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return (100 * v).toFixed(digits === undefined ? 2 : digits) + "%";
  }

  function buildTable(container, columns, rows) {
    container.innerHTML = "";
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const col of columns) {
      const th = document.createElement("th");
      th.textContent = col.label;
      if (col.title) th.title = col.title;
      if (col.numeric) th.className = "numeric";
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    for (const row of rows) {
      const tr = document.createElement("tr");
      for (const col of columns) {
        const td = document.createElement("td");
        td.textContent = col.get(row);
        if (col.numeric) td.className = "numeric";
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    container.appendChild(table);
  }

  function renderZipfProfile() {
    const note = document.getElementById("zipf-alloc-note");
    const alloc = zipf.allocation || [];
    if (alloc.length) {
      const top = alloc
        .filter((a) => !a.script.startsWith("<"))
        .slice(0, 6)
        .map((a) => `${a.script} ${pct(a.share)}`)
        .join(" · ");
      const frag = alloc.find((a) => a.script === "<byte_fragment>");
      note.textContent =
        "Static allocation across o200k_base mergeable ranks — " +
        top +
        (frag
          ? ` · ${frag.n_tokens.toLocaleString()} partial-UTF-8 byte fragments`
          : "");
    }

    buildTable(
      document.getElementById("zipf-profile-table"),
      [
        { label: "Language", get: (r) => `${r.name} (${r.code})` },
        { label: "Script", get: (r) => r.script || "—" },
        {
          label: "Tokens",
          numeric: true,
          title: "Corpus token count over FLORES dev+devtest",
          get: (r) => r.n_tokens.toLocaleString(),
        },
        {
          label: "Active types",
          numeric: true,
          title: "Distinct token ids the language reaches",
          get: (r) => r.n_types.toLocaleString(),
        },
        {
          label: "Share of vocab",
          numeric: true,
          title: "Active types as a share of the 199,998 mergeable ranks",
          get: (r) => pct(r.share_of_vocab, 3),
        },
        {
          label: "Byte-fragment mass",
          numeric: true,
          title:
            "Share of this language's tokens that are partial-UTF-8 byte fragments",
          get: (r) => pct(r.share_fragment_mass, 1),
        },
        {
          label: "Exclusive mass",
          numeric: true,
          title:
            "Share of token occurrences on types English never reaches (mass-weighted)",
          get: (r) => pct(r.share_mass_not_in_control, 1),
        },
        {
          label: "Whole-word coverage",
          numeric: true,
          title:
            "Share of distinct word types that encode to exactly one token (type-level)",
          get: (r) => pct(r.whole_word_coverage, 2),
        },
      ],
      zipf.profile || []
    );
  }

  function renderZipfMethod() {
    const container = document.getElementById("zipf-method");
    if (!container) return;
    container.innerHTML = "";
    for (const [i, step] of (zipf.method || []).entries()) {
      const details = document.createElement("details");
      details.className = "method-step";
      // First step open so the section does not read as an empty stack.
      if (i === 0) details.open = true;
      const summary = document.createElement("summary");
      summary.textContent = step.title;
      const body = document.createElement("div");
      body.className = "method-body";
      body.textContent = step.body;
      details.appendChild(summary);
      details.appendChild(body);
      container.appendChild(details);
    }
  }

  function renderZipfFigures() {
    const container = document.getElementById("zipf-figures");
    container.innerHTML = "";
    for (const fig of zipf.figures || []) {
      const wrap = document.createElement("figure");
      wrap.className = "zipf-figure";
      const img = document.createElement("img");
      img.src = fig.file;
      img.alt = fig.caption;
      img.loading = "lazy";
      const cap = document.createElement("figcaption");
      cap.textContent = fig.caption;
      wrap.appendChild(img);
      wrap.appendChild(cap);
      container.appendChild(wrap);
    }
  }

  function renderZipfHeatmap() {
    const view = zipfViewSelect.value;
    const metric = zipf.metrics.find((m) => m.id === zipfMetricSelect.value);
    const viewMeta = zipf.views.find((v) => v.id === view);

    zipfViewExplanation.textContent = viewMeta ? viewMeta.explanation : "";
    zipfMetricExplanation.textContent =
      metric.explanation + (metric.formula ? "\n\nComputed as:  " + metric.formula : "");

    const rows = zipf.rows.filter((r) => r.view === view);
    const lookup = new Map();
    for (const row of rows) {
      lookup.set(row.tokenizer_id + "|" + row.language, row);
    }
    const getValue = (metricId, tokId, langCode) => {
      const row = lookup.get(tokId + "|" + langCode);
      if (!row) return null;
      const v = row[metricId];
      return v === null || v === undefined ? null : v;
    };

    renderHeatmap(
      zipfHeatmap,
      zipfHeatmapTitle,
      metric,
      zipf.languages,
      zipf.tokenizers,
      getValue
    );

    // Append bootstrap intervals to the cell tooltips renderHeatmap produced.
    const cells = zipfHeatmap.querySelectorAll(".cell");
    cells.forEach((cell) => {
      const parts = (cell.title || "").split(" · ");
      if (parts.length < 2) return;
      const langName = parts[0];
      const lang = zipf.languages.find((l) => l.name === langName);
      const tokLabel = parts[1].split(":")[0];
      const tok = zipf.tokenizers.find((t) => t.label === tokLabel);
      if (!lang || !tok) return;
      const row = lookup.get(tok.id + "|" + lang.code);
      if (!row) return;
      const lo = row[metric.id + "_lo"];
      const hi = row[metric.id + "_hi"];
      const extras = [];
      if (lo !== undefined && lo !== null && hi !== undefined && hi !== null) {
        extras.push(`95% CI [${Number(lo).toFixed(3)}, ${Number(hi).toFixed(3)}]`);
      }
      if (row.n_types !== undefined && row.n_types !== null) {
        extras.push(`support V=${Math.round(row.n_types).toLocaleString()}`);
      }
      if (extras.length) cell.title = cell.title + " · " + extras.join(" · ");
    });
  }

  function setupZipf() {
    if (!zipf || !zipf.rows || !zipf.rows.length) {
      if (zipfMissing) zipfMissing.hidden = false;
      return;
    }
    zipfBody.hidden = false;

    const budgets = zipf.budgets || {};
    const budgetText = Object.keys(budgets).length
      ? " · matched budgets: " +
        Object.entries(budgets)
          .map(([unit, n]) => `${unit} ${Number(n).toLocaleString()}`)
          .join(", ")
      : "";
    document.getElementById("zipf-intro").textContent =
      zipf.explanation +
      "\n\n" +
      zipf.source +
      budgetText +
      (zipf.n_bootstrap ? ` · ${zipf.n_bootstrap} bootstrap draws` : "");

    for (const v of zipf.views) {
      const opt = document.createElement("option");
      opt.value = v.id;
      opt.textContent = v.label;
      zipfViewSelect.appendChild(opt);
    }
    zipfViewSelect.value = zipf.views[0].id;

    for (const m of zipf.metrics) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      zipfMetricSelect.appendChild(opt);
    }
    zipfMetricSelect.value = zipf.metrics[0].id;

    zipfViewSelect.addEventListener("change", renderZipfHeatmap);
    zipfMetricSelect.addEventListener("change", renderZipfHeatmap);

    renderZipfMethod();
    renderZipfProfile();
    renderZipfFigures();
  }

  const panels = {
    production: tabProduction,
    experiment: tabExperiment,
    zipf: document.getElementById("tab-zipf"),
  };

  function activateTab(name) {
    for (const [key, panel] of Object.entries(panels)) {
      if (!panel) continue;
      const active = key === name;
      panel.hidden = !active;
      panel.classList.toggle("active", active);
    }

    tabs.forEach((btn) => {
      const active = btn.dataset.tab === name;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });

    if (name === "production") {
      renderProduction();
    } else if (name === "zipf" && zipf && zipf.rows && zipf.rows.length) {
      renderZipfHeatmap();
    }
  }

  prodSelect.addEventListener("change", renderProduction);

  tabs.forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });

  buildExperimentSections();
  setupZipf();
  activateTab("production");
})();
