// Shared helpers for ScriptInsight pages: API access, escaping, theme, tabs and small SVG charts.
const SI = (() => {
  const TRAITS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'];

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  const pct = (value, digits = 1) => (value === null || value === undefined || Number.isNaN(value))
    ? '—' : `${(value * 100).toFixed(digits)}%`;

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  async function errorMessage(response) {
    try {
      const data = await response.json();
      if (data && data.error) return data.error;
    } catch (_) {
      // Non-JSON error body (e.g. a proxy or HTML error page).
    }
    if (response.status === 413) return 'Image is too large. Please select an image smaller than 16MB.';
    return `The server returned an error (HTTP ${response.status}). Please try again.`;
  }

  async function getJSON(url) {
    let response;
    try {
      response = await fetch(url, { headers: { Accept: 'application/json' } });
    } catch (_) {
      throw new Error('Could not reach the server. Make sure the app is running and try again.');
    }
    if (!response.ok && response.status !== 503) throw new Error(await errorMessage(response));
    return response.json();
  }

  // Line chart: series = [{label, points: [{x, y, flag?}], className}]
  function lineChart(series, { xLabel = 'epoch', yLabel = 'loss', height = 220 } = {}) {
    const all = series.flatMap((s) => s.points).filter((p) => Number.isFinite(p.y));
    if (!all.length) return '<p class="empty-note">No data points.</p>';
    const width = 640, pad = { l: 48, r: 14, t: 12, b: 34 };
    const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
    const sx = (x) => pad.l + ((x - xMin) / Math.max(xMax - xMin, 1)) * (width - pad.l - pad.r);
    const sy = (y) => pad.t + (1 - (y - yMin) / Math.max(yMax - yMin, 1e-9)) * (height - pad.t - pad.b);
    const ticks = [0, 0.5, 1].map((f) => yMin + f * (yMax - yMin));
    const lines = series.map((s) => {
      const pts = s.points.filter((p) => Number.isFinite(p.y));
      const path = pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
      const flags = pts.filter((p) => p.flag).map((p) =>
        `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="3" class="chart-flag"><title>${escapeHtml(p.flag)}</title></circle>`).join('');
      return `<path d="${path}" class="chart-line ${s.className}" fill="none"/>${flags}`;
    }).join('');
    const legend = series.map((s) => `<span class="legend-item"><i class="legend-swatch ${s.className}"></i>${escapeHtml(s.label)}</span>`).join('');
    return `
      <svg class="line-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(series.map((s) => s.label).join(' and '))} by ${xLabel}">
        ${ticks.map((t) => `<line x1="${pad.l}" x2="${width - pad.r}" y1="${sy(t)}" y2="${sy(t)}" class="chart-grid"/>
          <text x="${pad.l - 6}" y="${sy(t) + 4}" text-anchor="end" class="chart-tick">${t.toFixed(2)}</text>`).join('')}
        <text x="${width / 2}" y="${height - 6}" text-anchor="middle" class="chart-tick">${xLabel}</text>
        <text x="12" y="${height / 2}" text-anchor="middle" class="chart-tick" transform="rotate(-90 12 ${height / 2})">${yLabel}</text>
        ${lines}
      </svg>
      <div class="chart-legend">${legend}</div>`;
  }

  function confusionTable(matrix, labels, caption) {
    const max = Math.max(1, ...matrix.flat());
    return `
      <table class="confusion"><caption>${escapeHtml(caption)}</caption>
        <thead><tr><th scope="col">True ↓ / Predicted →</th>${labels.map((l) => `<th scope="col">${escapeHtml(l)}</th>`).join('')}</tr></thead>
        <tbody>${matrix.map((row, i) => `<tr><th scope="row">${escapeHtml(labels[i])}</th>${row.map((v) =>
          `<td style="--heat:${(v / max).toFixed(2)}">${v}</td>`).join('')}</tr>`).join('')}</tbody>
      </table>`;
  }

  function initTheme(toggle, icon) {
    const apply = (theme) => {
      document.documentElement.setAttribute('data-theme', theme);
      try { localStorage.setItem('theme', theme); } catch (_) { /* storage blocked */ }
      icon.textContent = theme === 'dark' ? '☀️' : '🌙';
    };
    let saved = null;
    try { saved = localStorage.getItem('theme'); } catch (_) { /* storage blocked */ }
    apply(saved || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
    toggle.addEventListener('click', () => {
      apply(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
    });
  }

  const tabListeners = [];
  function showTab(name, { focus = false } = {}) {
    document.querySelectorAll('[data-tab]').forEach((tab) => {
      const active = tab.dataset.tab === name;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      if (active && focus) tab.focus();
    });
    document.querySelectorAll('[data-panel]').forEach((panel) => {
      panel.classList.toggle('hidden', panel.dataset.panel !== name);
    });
    if (location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);
    tabListeners.forEach((fn) => fn(name));
  }

  function initTabs() {
    const tabs = [...document.querySelectorAll('[data-tab]')];
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => showTab(tab.dataset.tab));
      tab.addEventListener('keydown', (event) => {
        if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
        const next = tabs[(index + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
        showTab(next.dataset.tab, { focus: true });
      });
    });
    const initial = location.hash.slice(1);
    showTab(tabs.some((t) => t.dataset.tab === initial) ? initial : tabs[0].dataset.tab);
  }

  return {
    TRAITS, escapeHtml, pct, formatBytes, errorMessage, getJSON, lineChart, confusionTable, initTheme,
    initTabs, showTab, onTab: (fn) => tabListeners.push(fn)
  };
})();
