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

  const expExplanation = document.getElementById("experiment-explanation");
  const expPlot = document.getElementById("experiment-plot");
  const expSelect = document.getElementById("experiment-metric-select");
  const expMetricExplanation = document.getElementById("experiment-metric-explanation");
  const expHeatmap = document.getElementById("experiment-heatmap");
  const expHeatmapTitle = document.getElementById("experiment-heatmap-title");

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
  if (data.experiment) {
    fillMetricSelect(expSelect);
    expExplanation.textContent = data.experiment.explanation;
    if (data.experiment.plot) {
      expPlot.src = data.experiment.plot;
    }
  }

  const prodLookup = new Map();
  for (const row of data.rows) {
    prodLookup.set(row.tokenizer_id + "|" + row.language, row);
  }

  let expLookup = null;
  if (data.experiment) {
    expLookup = new Map();
    for (const row of data.experiment.rows) {
      expLookup.set(row.tokenizer_id + "|" + row.language, row);
    }
  }

  function getProdValue(metricId, tokId, langCode) {
    const row = prodLookup.get(tokId + "|" + langCode);
    if (!row) return null;
    const v = row[metricId];
    return v === null || v === undefined ? null : v;
  }

  function getExpValue(metricId, armId, langCode) {
    if (!expLookup) return null;
    const row = expLookup.get(armId + "|" + langCode);
    if (!row) return null;
    const v = row[metricId];
    return v === null || v === undefined ? null : v;
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

  function renderExperiment() {
    if (!data.experiment) return;
    const metric = data.metrics.find((m) => m.id === expSelect.value);
    expMetricExplanation.textContent = metric.explanation;
    renderHeatmap(
      expHeatmap,
      expHeatmapTitle,
      metric,
      data.experiment.languages,
      data.experiment.arms,
      getExpValue
    );
  }

  function activateTab(name) {
    const isProduction = name === "production";
    tabProduction.hidden = !isProduction;
    tabProduction.classList.toggle("active", isProduction);
    tabExperiment.hidden = isProduction;
    tabExperiment.classList.toggle("active", !isProduction);

    tabs.forEach((btn) => {
      const active = btn.dataset.tab === name;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });

    if (isProduction) {
      renderProduction();
    } else {
      renderExperiment();
    }
  }

  prodSelect.addEventListener("change", renderProduction);
  if (data.experiment) {
    expSelect.addEventListener("change", renderExperiment);
  }

  tabs.forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });

  activateTab("production");
})();
