// Dashboard and Model Performance tabs. Every number shown comes from /api/info, /api/metrics or this
// tab's own session history; missing data is shown as unavailable rather than estimated.
(() => {
  const dashboardContent = document.getElementById('dashboardContent');
  const evaluationContent = document.getElementById('evaluationContent');
  const { escapeHtml: esc, num, signed, pct, modelName } = SI;
  let loaded = null;

  const STATES = {
    loaded: ['chip-ok', 'Ready'],
    not_trained: ['chip-muted', 'Not installed'],
    error: ['chip-muted', 'Unavailable'],
    disabled: ['chip-muted', 'Turned off'],
  };
  const TASKS = {
    regression: 'Scores for all five traits',
    multiclass: 'One of five trait categories',
    multilabel: 'High or low for each trait',
  };
  const SERIES = ['fill-base', 'fill-1', 'fill-2', 'fill-3', 'fill-4'];

  const chip = (cls, label) => `<span class="chip ${cls}">${esc(label)}</span>`;
  const tile = (label, value, note) => `<div class="stat-tile"><span class="stat-label">${esc(label)}</span>
    <strong>${esc(value)}</strong>${note ? `<span class="stat-note">${esc(note)}</span>` : ''}</div>`;

  function sessionHistory() {
    try { return JSON.parse(sessionStorage.getItem('scriptinsight-history') || '[]'); } catch (_) { return []; }
  }

  function renderDashboard({ info, metrics }) {
    const h = metrics.hienwrite || {};
    const counts = h.validation ? h.validation.counts : null;
    const split = h.split;
    const ready = info.models.filter((m) => m.available);
    const others = info.models.filter((m) => !m.available);
    const history = sessionHistory();

    const tiles = [
      tile('Models ready', String(ready.length)),
      counts ? tile('Training images', String(counts.usable_samples), `from ${counts.participants_usable} writers`) : '',
      split ? tile('Test images', String(split.counts.test.samples), 'held out for evaluation') : '',
      tile('Analyses this session', String(history.length), 'in this browser tab'),
    ].join('');

    const modelItems = ready.map((m) => `
      <li><div class="model-row"><strong>${esc(modelName(m.model, m.display_name))}</strong>
        <span class="chip-group">${m.model === info.primary_model ? chip('chip-primary', 'Default') : ''}${chip(...STATES.loaded)}</span></div>
        <span class="muted">${esc(TASKS[m.task] || 'Model available')}</span></li>`).join('');
    const modelsCard = `
      <div class="dash-card">
        <h3>Available models</h3>
        ${ready.length ? `<ul class="model-list">${modelItems}</ul>`
          : '<p class="empty-note">No prediction model is installed. Predictions are unavailable until a trained model is added.</p>'}
        ${others.length ? `<p class="muted small-print">Not installed here: ${others.map((m) => esc(modelName(m.model, m.display_name))).join(', ')}.</p>` : ''}
      </div>`;

    const howCard = `
      <div class="dash-card">
        <h3>How it works</h3>
        <ol class="how-list">
          <li><strong>Upload</strong> a photo or scan of handwriting.</li>
          <li><strong>Preprocess</strong> the image to remove noise and standardize the sample.</li>
          <li><strong>Estimate</strong> Big Five trait scores or trait categories using the selected model.</li>
        </ol>
      </div>`;

    const dataCard = counts && split ? `
      <div class="dash-card">
        <h3>Training data</h3>
        <p>HiEnWrite handwriting samples paired with self-reported Big Five questionnaire scores.</p>
        <dl class="facts">
          <div><dt>Images used</dt><dd>${counts.usable_samples}</dd></div>
          <div><dt>Writers</dt><dd>${counts.participants_usable}</dd></div>
          <div><dt>Training</dt><dd>${split.counts.train.samples}</dd></div>
          <div><dt>Validation</dt><dd>${split.counts.val.samples}</dd></div>
          <div><dt>Test</dt><dd>${split.counts.test.samples}</dd></div>
        </dl>
        <p class="muted small-print">Checker, Yadav &amp; Katarya (2025), CC BY-NC 4.0.</p>
      </div>` : `
      <div class="dash-card"><h3>Training data</h3><p class="empty-note">A summary of the training data is not available.</p></div>`;

    const recent = history.slice(0, 3).map((item) => `
      <li><strong>${esc(item.file.name)}</strong>
        <span class="muted">${esc(modelName(item.data.prediction.model, item.data.prediction.display_name))} · ${esc(new Date(item.time).toLocaleTimeString())}</span></li>`).join('');
    const activityCard = `
      <div class="dash-card">
        <h3>Recent activity</h3>
        ${recent ? `<ul class="plain-list">${recent}</ul>` : '<p class="empty-note">No analyses in this browser tab yet.</p>'}
        <button type="button" class="secondary-btn small" data-goto="${recent ? 'history' : 'analyze'}">${recent ? 'View history' : 'Analyze a sample'}</button>
      </div>`;

    dashboardContent.innerHTML = `<div class="stat-row">${tiles}</div>
      <div class="dashboard-grid">${modelsCard}${howCard}${dataCard}${activityCard}</div>`;
  }

  function verdict(interval) {
    if (!interval) return '—';
    if (interval[0] > 0) return chip('chip-warn', 'Higher error than baseline');
    if (interval[1] < 0) return chip('chip-ok', 'Lower error than baseline');
    return chip('chip-muted', 'No clear difference');
  }

  function performanceSection(e, history) {
    const traits = SI.TRAITS.filter((t) => t in e.baseline.test_metrics.per_trait);
    const models = Object.entries(e.models)
      .sort(([, a], [, b]) => a.test_metrics.macro.rmse - b.test_metrics.macro.rmse);
    const boot = e.bootstrap.models;
    const better = models.filter(([n]) => boot[n] && boot[n].macro.rmse_minus_baseline_95ci[1] < 0).length;
    const summary = better
      ? `${better} of ${models.length} models predicted scores with lower error than the baseline on this test set.`
      : `None of the ${models.length} models predicted scores more accurately than the baseline on this test set.`;

    const row = (label, m, extra, cls = '') => `<tr class="${cls}"><th scope="row">${esc(label)}</th>
      <td class="num">${num(m.mae)}</td><td class="num">${num(m.rmse)}</td><td class="num">${signed(m.r2)}</td>
      <td class="num">${m.pearson_r === null ? '—' : signed(m.pearson_r)}</td><td>${extra}</td></tr>`;
    const overall = `
      <div class="table-wrap"><table class="data-table">
        <caption>Average over the five traits</caption>
        <thead><tr><th scope="col">Model</th><th scope="col" class="num">MAE</th><th scope="col" class="num">RMSE</th>
          <th scope="col" class="num">R²</th><th scope="col" class="num">Pearson r</th><th scope="col">Compared with baseline</th></tr></thead>
        <tbody>
          ${row(modelName('mean_baseline'), e.baseline.test_metrics.macro, '<span class="muted">Reference</span>', 'baseline-row')}
          ${models.map(([n, m]) => row(modelName(n, m.display_name), m.test_metrics.macro,
            verdict(boot[n] && boot[n].macro.rmse_minus_baseline_95ci))).join('')}
        </tbody></table></div>`;

    const series = [{ label: modelName('mean_baseline'), entry: e.baseline }, ...models.map(([n, m]) => ({ label: modelName(n, m.display_name), entry: m }))]
      .map((s, i) => ({ label: s.label, className: SERIES[i] || 'fill-4',
        values: traits.map((t) => s.entry.test_metrics.per_trait[t].rmse) }));
    const chart = SI.barChart(traits, series, { yLabel: 'RMSE (score points)' });

    const cell = (m) => `${num(m.rmse)}<span class="cell-sub">r ${m.pearson_r === null ? '—' : signed(m.pearson_r)}</span>`;
    const perTrait = `
      <div class="table-wrap"><table class="data-table">
        <caption>RMSE for each trait (lower is better), with the correlation r between predicted and self-reported scores</caption>
        <thead><tr><th scope="col">Trait</th><th scope="col" class="num">Images</th><th scope="col" class="num">Baseline</th>
          ${models.map(([n, m]) => `<th scope="col" class="num" title="${esc(modelName(n, m.display_name))}">${esc(SI.modelShort(n, m.display_name))}</th>`).join('')}</tr></thead>
        <tbody>${traits.map((t) => `<tr><th scope="row">${esc(t)}</th>
          <td class="num">${e.baseline.test_metrics.per_trait[t].n}</td>
          <td class="num">${num(e.baseline.test_metrics.per_trait[t].rmse)}</td>
          ${models.map(([, m]) => `<td class="num">${cell(m.test_metrics.per_trait[t])}</td>`).join('')}</tr>`).join('')}
        </tbody></table></div>`;

    const masked = (e.split.masked_targets || []).map((m) =>
      `<li>${esc(m.target)} is evaluated on ${m.language === 'h' ? 'English' : 'Hindi'} samples only (${e.n_test_valid_per_trait[m.target]} test images): the source data gives every ${m.language === 'h' ? 'Hindi' : 'English'} sample the same ${esc(m.target)} score, so those labels were not used.</li>`).join('');

    const curve = history ? `
      <details class="disclosure">
        <summary>Training curve: ResNet18</summary>
        <p class="muted small-print">Loss on the training and validation images after each epoch. The model saved at the marked epoch is the one used by the app.</p>
        ${SI.lineChart([
          { label: 'Training loss', className: 'series-a', points: history.epochs.map((x) => ({ x: x.epoch, y: x.train_loss })) },
          { label: 'Validation loss', className: 'series-b', points: history.epochs.map((x) => ({ x: x.epoch, y: x.val_loss })) },
        ], { xLabel: 'Epoch', yLabel: 'Loss', markX: history.best_epoch, markLabel: 'saved' })}
      </details>` : '';

    return `
      <p>Each model was tested once on <strong>${e.n_test} handwriting images from ${e.n_test_groups} writers</strong> that were not
        used for training or for choosing model settings. Scores are on the questionnaire scale used by the dataset. The
        <strong>baseline</strong> predicts the training-data average for every image; a useful model should have lower error than it.</p>
      <p class="summary-callout">${esc(summary)}</p>
      ${overall}
      <h3 class="section-title">Error by trait</h3>
      ${chart}
      ${perTrait}
      <h3 class="section-title">Reading these results</h3>
      <ul class="plain-list bullets">
        <li><strong>MAE</strong> and <strong>RMSE</strong> are the typical size of the prediction error in score points; RMSE weighs large errors more.</li>
        <li><strong>R²</strong> below zero means the errors are larger than those of always predicting the test-set average.</li>
        <li><strong>Pearson r</strong> shows whether predictions rise and fall with the real scores (0 = no relationship). It is undefined for the baseline, which always predicts the same value.</li>
        <li>"Compared with baseline" uses a 95% bootstrap interval over writers; "No clear difference" means the interval includes zero.</li>
        ${masked}
      </ul>
      <p class="muted small-print">Performance is based on the evaluated dataset and test split; results may not generalize to other handwriting samples.</p>
      ${curve}`;
  }

  function classifierSection(evaluation) {
    const rows = Object.entries(evaluation.models).map(([n, e]) => {
      const t = e.test_metrics;
      const acc = t.task === 'multiclass' ? pct(t.accuracy) : `${pct(t.macro.accuracy)} (mean of traits)`;
      return `<tr><th scope="row">${esc(modelName(n, e.display_name))}</th><td class="num">${acc}</td><td class="num">${num(t.macro.precision, 3)}</td>
        <td class="num">${num(t.macro.recall, 3)}</td><td class="num">${num(t.macro.f1, 3)}</td></tr>`;
    }).join('');
    const base = evaluation.baseline_majority_class;
    const baseRow = base ? `<tr class="baseline-row"><th scope="row">Baseline: always predict ${esc(base.class)}</th>
        <td class="num">${pct(base.accuracy)}</td><td class="num">—</td><td class="num">—</td><td class="num">${num(base.macro_f1, 3)}</td></tr>` : '';
    const data = evaluation.dataset || {};
    const notes = [...(data.warnings || []), ...(evaluation.warnings || []).filter((w) => !w.startsWith('Majority-class baseline'))];
    const noteHtml = notes.length ? `<div class="summary-callout"><ul class="plain-list bullets">${notes.map((w) => `<li>${esc(w)}</li>`).join('')}</ul></div>` : '';
    const copies = data.n_groups && data.n_samples && data.n_groups < data.n_samples
      ? `<p class="muted small-print">The ${data.n_samples} images are copies of ${data.n_groups} original photos, so these scores rest on far fewer independent test samples than n suggests: one photo can move accuracy by several percentage points. Compare each model with the baseline row; differences of a few points are within noise.</p>` : '';
    const plots = (evaluation.plots || []).filter((f) => f.startsWith('confusion_')).map((f) =>
      `<img src="/artifacts/evaluation/${esc(f)}" alt="Confusion matrix: ${esc(f.replace(/^confusion_|\.png$/g, ''))}" loading="lazy" style="max-width:100%;height:auto">`).join('');
    const plotHtml = plots ? `<details class="disclosure"><summary>Confusion matrices (test images)</summary>${plots}</details>` : '';
    return `<h3 class="section-title">Classification models</h3>
      ${noteHtml}
      <div class="table-wrap"><table class="data-table"><caption>Held-out test images (n = ${evaluation.n_test}); macro averages</caption>
        <thead><tr><th scope="col">Model</th><th scope="col" class="num">Accuracy</th><th scope="col" class="num">Precision</th><th scope="col" class="num">Recall</th><th scope="col" class="num">F1</th></tr></thead>
        <tbody>${baseRow}${rows}</tbody></table></div>
      ${copies}${plotHtml}`;
  }

  function legacySection(metrics) {
    const legacy = metrics.legacy_recorded;
    const history = metrics.training_history.legacy_resnet50;
    if (!legacy || !history) return '';
    let x = 0;
    const points = history.epochs.map((e) => ({ ...e, x: ++x }));
    const perClass = Object.entries(legacy.test_metrics_recorded.per_class).map(([name, m]) =>
      `<tr><th scope="row">${esc(name)}</th><td class="num">${num(m.precision)}</td><td class="num">${num(m.recall)}</td><td class="num">${num(m.f1)}</td><td class="num">${m.support}</td></tr>`).join('');
    return `
      <details class="disclosure">
        <summary>Earlier prototype: ResNet50 category classifier</summary>
        <p>The first version of the project used a ResNet50 that picks <em>one</em> of five trait categories per image. Its results below were recorded in the original training notebook. It was trained on images that are now part of the new test split, so it cannot be fairly re-scored on it. They are not comparable with the models above (different split and different training data handling).</p>
        <div class="stat-row compact">
          ${tile('Recorded test accuracy', pct(legacy.test_metrics_recorded.accuracy), 'chance is about 20%')}
          ${tile('Weighted F1', num(legacy.test_metrics_recorded.weighted_f1, 3))}
        </div>
        <div class="table-wrap"><table class="data-table"><caption>Recorded results per category (n = 660 test images)</caption>
          <thead><tr><th scope="col">Category</th><th scope="col" class="num">Precision</th><th scope="col" class="num">Recall</th><th scope="col" class="num">F1</th><th scope="col" class="num">Images</th></tr></thead>
          <tbody>${perClass}</tbody></table></div>
        ${SI.lineChart([
          { label: 'Training loss', className: 'series-a', points: points.map((e) => ({ x: e.x, y: e.train_loss })) },
          { label: 'Validation loss', className: 'series-b', points: points.map((e) => ({ x: e.x, y: e.val_loss })) },
        ], { xLabel: 'Logged epoch', yLabel: 'Loss' })}
        <p class="muted small-print">Known issues with this earlier model:</p>
        <ul class="plain-list bullets small-print">${legacy.known_issues.map((i) => `<li>${esc(i)}</li>`).join('')}
          <li>The project report's 85% accuracy target applies to classification models with per-trait labels; it could not be measured.</li></ul>
      </details>`;
  }

  function renderEvaluation({ metrics }) {
    const h = metrics.hienwrite || {};
    let html = h.evaluation_available
      ? performanceSection(h.evaluation, h.training_history)
      : `<div class="empty-state"><p><strong>No evaluation results are available yet.</strong></p>
          <p>Results appear here after the models have been evaluated on held-out test images.</p></div>`;
    if (metrics.evaluation && Object.keys(metrics.evaluation.models).length) html += classifierSection(metrics.evaluation);
    evaluationContent.innerHTML = html + legacySection(metrics);
  }

  async function load() {
    if (loaded) return loaded;
    loaded = Promise.all([SI.getJSON('/api/info'), SI.getJSON('/api/metrics')])
      .then(([info, metrics]) => ({ info, metrics }))
      .catch((error) => { loaded = null; throw error; });
    return loaded;
  }

  async function show(tab) {
    if (tab !== 'dashboard' && tab !== 'evaluation') return;
    const target = tab === 'dashboard' ? dashboardContent : evaluationContent;
    try {
      const data = await load();
      if (tab === 'dashboard') renderDashboard(data);
      else renderEvaluation(data);
    } catch (error) {
      target.innerHTML = `<p class="file-message error">${esc(error.message)}</p>
        <button type="button" class="secondary-btn" data-retry="${tab}">Retry</button>`;
    }
  }

  document.addEventListener('click', (event) => {
    const retry = event.target.closest('[data-retry]');
    if (retry) show(retry.dataset.retry);
    const goto = event.target.closest('#dashboardContent [data-goto]');
    if (goto) SI.showTab(goto.dataset.goto);
  });
  SI.onTab(show);
  show(location.hash.slice(1));
})();
