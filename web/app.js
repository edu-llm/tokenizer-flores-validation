(function () {
  const data = window.METRICS_DATA;
  if (!data) {
    document.body.innerHTML = "<p>Missing data.js — run <code>python scripts/export_web_data.py</code>.</p>";
    return;
  }

  const select = document.getElementById("metric-select");
  const explanation = document.getElementById("explanation");
  const table = document.getElementById("metrics-table");
  const heatmap = document.getElementById("heatmap");
  const caption = document.getElementById("caption");
  const tableTitle = document.getElementById("table-title");
  const heatmapTitle = document.getElementById("heatmap-title");

  caption.textContent =
    data.source +
    " · " +
    data.n_sentences +
    " sentences · " +
    data.languages.length +
    " languages · " +
    data.tokenizers.length +
    " tokenizers";

  for (const m of data.metrics) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    select.appendChild(opt);
  }

  const lookup = new Map();
  for (const row of data.rows) {
    lookup.set(row.tokenizer_id + "|" + row.language, row);
  }

  function getValue(metricId, tokId, langCode) {
    const row = lookup.get(tokId + "|" + langCode);
    if (!row) return null;
    const v = row[metricId];
    return v === null || v === undefined ? null : v;
  }

  function formatValue(v) {
    if (v === null || Number.isNaN(v)) return "—";
    return Number(v).toFixed(3);
  }

  function severity(metric, values) {
    // Map each numeric value to [0,1] where 1 = worst.
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
    // Pale cream → deep red
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

  /** Higher = worse. Uses raw means (not per-slice normalization). */
  function badnessScore(metric, values) {
    const m = meanNumeric(values);
    if (m === null) return Number.NEGATIVE_INFINITY; // missing → sort last
    return metric.higher_is_worse ? m : -m;
  }

  function applicableLanguages(metric) {
    const omit = new Set(metric.omit_languages || []);
    return data.languages.filter((lang) => {
      if (omit.has(lang.code)) return false;
      // Also drop languages with no numeric values for this metric.
      return data.tokenizers.some((t) => getValue(metric.id, t.id, lang.code) !== null);
    });
  }

  function sortedAxes(metric) {
    const langsAvail = applicableLanguages(metric);
    // Tokenizers: worst → best (left → right)
    const toks = [...data.tokenizers].sort((a, b) => {
      const av = langsAvail.map((l) => getValue(metric.id, a.id, l.code));
      const bv = langsAvail.map((l) => getValue(metric.id, b.id, l.code));
      return badnessScore(metric, bv) - badnessScore(metric, av);
    });
    // Languages: worst → best (top → bottom)
    const langs = [...langsAvail].sort((a, b) => {
      const av = toks.map((t) => getValue(metric.id, t.id, a.code));
      const bv = toks.map((t) => getValue(metric.id, t.id, b.code));
      return badnessScore(metric, bv) - badnessScore(metric, av);
    });
    return { toks, langs };
  }

  function render() {
    const metric = data.metrics.find((m) => m.id === select.value);
    explanation.textContent = metric.explanation;
    tableTitle.textContent = metric.label + " — table (worst → best)";
    heatmapTitle.textContent = metric.label + " — heatmap (worst → best; redder = worse)";

    const { toks, langs } = sortedAxes(metric);

    // Table
    let thead = "<thead><tr><th>Language</th>";
    for (const t of toks) thead += `<th>${t.label}</th>`;
    thead += "</tr></thead>";

    let tbody = "<tbody>";
    for (const lang of langs) {
      tbody += `<tr><td>${lang.name}<br><span style="color:#57534e;font-size:0.75rem">${lang.code}</span></td>`;
      for (const t of toks) {
        const v = getValue(metric.id, t.id, lang.code);
        const cls = v === null ? ' class="na"' : "";
        tbody += `<td${cls}>${formatValue(v)}</td>`;
      }
      tbody += "</tr>";
    }
    tbody += "</tbody>";
    table.innerHTML = thead + tbody;

    // Heatmap matrix (same order as table)
    const flat = [];
    for (const lang of langs) {
      for (const t of toks) {
        flat.push(getValue(metric.id, t.id, lang.code));
      }
    }
    const sevs = severity(metric, flat);

    heatmap.style.gridTemplateColumns = `minmax(6.5rem, 9rem) repeat(${toks.length}, minmax(3.4rem, 1fr))`;
    heatmap.innerHTML = "";

    const corner = document.createElement("div");
    corner.className = "cell corner";
    heatmap.appendChild(corner);
    for (const t of toks) {
      const h = document.createElement("div");
      h.className = "cell col-head";
      h.textContent = t.label;
      heatmap.appendChild(h);
    }

    let idx = 0;
    for (const lang of langs) {
      const rh = document.createElement("div");
      rh.className = "cell row-head";
      rh.textContent = lang.name;
      rh.title = lang.code;
      heatmap.appendChild(rh);
      for (const t of toks) {
        const v = flat[idx];
        const cell = document.createElement("div");
        cell.className = "cell";
        cell.style.background = heatColor(sevs[idx]);
        cell.textContent = formatValue(v);
        cell.title = `${lang.name} · ${t.label}: ${formatValue(v)}`;
        heatmap.appendChild(cell);
        idx += 1;
      }
    }
  }

  select.addEventListener("change", render);
  select.value = data.metrics[0].id;
  render();
})();
