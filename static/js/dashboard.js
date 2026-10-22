// Dashboard and Evaluation tabs. Every number shown comes from /api/info or /api/metrics;
// missing data is shown as unavailable rather than estimated.
(() => {
  const dashboardContent = document.getElementById('dashboardContent');
  const evaluationContent = document.getElementById('evaluationContent');
  const { escapeHtml: esc, pct } = SI;
  let loaded = null;

  const STATE_CHIPS = {
    loaded: ['chip-ok', 'Loaded'],
    not_trained: ['chip-muted', 'Not trained'],
    error: ['chip-error', 'Load error'],
    disabled: ['chip-muted', 'Disabled'],
  };

  function chip(state) {
    const [cls, label] = STATE_CHIPS[state] || ['chip-muted', state];
    return `<span class="chip ${cls}">${esc(label)}</span>`;
  }

  function stat(label, value, note) {
    return `<div class="stat"><span class="stat-label">${esc(label)}</span><strong>${esc(value)}</strong>${
      note ? `<span class="stat-note">${esc(note)}</span>` : ''}</div>`;
  }

  function provenance(kind) {
    const map = {
      measured: ['chip-ok', 'Measured by evaluate.py'],
      recorded: ['chip-warn', 'Recorded in notebook · not reproducible'],
      unavailable: ['chip-muted', 'Unavailable'],
    };
    const [cls, label] = map[kind];
    return `<span class="chip ${cls}">${label}</span>`;
  }

  function renderDashboard({ info, metrics }) {
    const models = info.models.map((m) => `
      <li><div class="model-row"><strong>${esc(m.display_name)}</strong>${chip(m.state)}</div>
        <span class="muted">${m.available
          ? `${esc(m.task === 'multilabel' ? 'Five independent binary traits' : 'One category out of five')} · ${esc((m.score_type || '').replace(/_/g, ' '))}`
          : esc(m.message || '')}</span></li>`).join('');

    const legacy = metrics.legacy_recorded;
    const legacyHistory = metrics.training_history.legacy_resnet50;
    const legacyCard = legacy ? `
      <div class="dash-card">
        <h4>Legacy ResNet50 results ${provenance('recorded')}</h4>
        <div class="stats">
          ${stat('Test accuracy', pct(legacy.test_metrics_recorded.accuracy), 'single-label, 5 categories')}
          ${stat('Weighted F1 (test)', legacy.test_metrics_recorded.weighted_f1?.toFixed(3) ?? '—')}
          ${stat('Best validation accuracy', pct(legacy.best_validation_accuracy_recorded), 'possible train/val leakage')}
          ${stat('Epochs logged', String(legacyHistory.epochs.length),
            `${legacyHistory.epochs.filter((e) => e.partial_epoch).length} ran only one batch`)}
        </div>
        <p class="muted small-print">Chance level for 5 balanced categories is about 20%.</p>
      </div>` : `<div class="dash-card"><h4>Legacy ResNet50 results</h4>${provenance('unavailable')}</div>`;

    const split = metrics.dataset_split;
    const datasetCard = split ? `
      <div class="dash-card">
        <h4>Dataset split ${provenance('measured')}</h4>
        <div class="stats">
          ${stat('Images', String(split.n_samples))}
          ${stat('Independent groups', String(split.n_groups), split.grouping)}
          ${stat('Train / val / test', ['train', 'val', 'test'].map((s) => split.distribution[s].n).join(' / '))}
          ${stat('Labels', split.label_mode === 'multilabel' ? 'Five binary traits' : 'One category per image')}
        </div>
        ${(split.warnings || []).map((w) => `<p class="note">${esc(w)}</p>`).join('')}
      </div>` : `
      <div class="dash-card">
        <h4>Dataset ${provenance('unavailable')}</h4>
        <p>No labelled dataset is configured, so the ResNet18, SVM and Random Forest models are not trained yet.</p>
        ${legacy ? `<p class="muted small-print">The legacy model's notebook recorded ${
          Object.values(legacy.dataset_recorded.train_folder_counts).reduce((a, b) => a + b, 0)} train-folder and ${
          Object.values(legacy.dataset_recorded.test_folder_counts).reduce((a, b) => a + b, 0)} test-folder images. Source and label provenance are undocumented, and the files are not available.</p>` : ''}
      </div>`;

    const evaluated = metrics.evaluation ? Object.entries(metrics.evaluation.models) : [];
    const targetRows = evaluated.filter(([, e]) => e.validation_accuracy_target.applicable).map(([, e]) =>
      Object.entries(e.validation_accuracy_target.traits).map(([trait, r]) =>
        `<li>${esc(e.display_name)} · ${esc(trait)}: ${pct(r.accuracy)} ${r.met ? '<span class="chip chip-ok">meets 85%</span>' : '<span class="chip chip-error">below 85%</span>'}</li>`).join('')).join('');
    const targetCard = `
      <div class="dash-card">
        <h4>Report target: 85% validation accuracy (Openness, Conscientiousness)</h4>
        ${targetRows ? `<ul class="plain-list">${targetRows}</ul>` : `<p>Not measurable yet. Per-trait accuracy needs a model trained on independent trait labels. The legacy model only outputs one category per image, so it has no per-trait accuracy.</p>`}
      </div>`;

    dashboardContent.innerHTML = `
      <div class="dash-card"><h4>Models</h4><ul class="model-list">${models}</ul>
        <p class="muted small-print">Primary model for predictions: ${esc(info.primary_model || 'none')}</p></div>
      ${legacyCard}${datasetCard}${targetCard}`;
  }

  function metricsTable(evaluation) {
    const rows = Object.values(evaluation.models).map((e) => {
      const t = e.test_metrics;
      const acc = t.task === 'multiclass' ? pct(t.accuracy) : `${pct(t.macro.accuracy)} (mean of traits)`;
      return `<tr><th scope="row">${esc(e.display_name)}</th><td>${acc}</td><td>${t.macro.precision.toFixed(3)}</td>
        <td>${t.macro.recall.toFixed(3)}</td><td>${t.macro.f1.toFixed(3)}</td><td>${t.latency_ms.median.toFixed(1)} ms</td></tr>`;
    }).join('');
    return `<div class="table-wrap"><table class="data-table"><caption>Held-out test split (n = ${evaluation.n_test}); macro averages</caption>
      <thead><tr><th scope="col">Model</th><th scope="col">Accuracy</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1</th><th scope="col">Median latency</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
  }

  function perModelDetails(name, e) {
    const t = e.test_metrics;
    const matrices = t.task === 'multiclass'
      ? SI.confusionTable(t.confusion_matrix, t.label_order, `${e.display_name}: test confusion matrix`)
      : Object.entries(t.per_trait).map(([trait, m]) =>
        SI.confusionTable(m.confusion_matrix, ['low', 'high'], `${trait}: accuracy ${pct(m.accuracy)}, F1 ${m.f1.toFixed(3)}`)).join('');
    const val = e.validation_metrics;
    return `<details class="model-details"><summary>${esc(e.display_name)} details</summary>
      <p class="muted">Artifact ${esc(e.artifact)} (sha256 ${esc(e.artifact_sha256_16)}) · score type ${esc(e.score_type)}</p>
      <h5>Validation (model selection) ${val ? `macro-F1 ${val.macro.f1.toFixed(3)}` : '—'}</h5>
      <h5>Test confusion matrices</h5><div class="matrix-grid">${matrices}</div></details>`;
  }

  function legacySection(metrics) {
    const legacy = metrics.legacy_recorded;
    const history = metrics.training_history.legacy_resnet50;
    if (!legacy || !history) return '';
    let x = 0;
    const points = history.epochs.map((e) => ({ ...e, x: ++x }));
    const flag = (e) => e.partial_epoch ? `epoch ${e.epoch} (${e.phase}) ran only one batch` : null;
    const perClass = Object.entries(legacy.test_metrics_recorded.per_class).map(([name, m]) =>
      `<tr><th scope="row">${esc(name)}</th><td>${m.precision.toFixed(2)}</td><td>${m.recall.toFixed(2)}</td><td>${m.f1.toFixed(2)}</td><td>${m.support}</td></tr>`).join('');
    return `
      <h4 class="section-title">Legacy ResNet50 ${provenance('recorded')}</h4>
      <p class="muted">${esc(legacy.provenance)} Its numbers are not comparable with the models above: different data, single-label task.</p>
      <div class="table-wrap"><table class="data-table"><caption>Recorded test classification report (n = 660)</caption>
        <thead><tr><th scope="col">Category</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1</th><th scope="col">Support</th></tr></thead>
        <tbody>${perClass}</tbody></table></div>
      <p class="muted small-print">${esc(legacy.test_metrics_recorded.note)}</p>
      <h5>Recorded training history (frozen-backbone phase, then fine-tune phase)</h5>
      ${SI.lineChart([
        { label: 'train loss', className: 'series-a', points: points.map((e) => ({ x: e.x, y: e.train_loss, flag: flag(e) })) },
        { label: 'validation loss', className: 'series-b', points: points.map((e) => ({ x: e.x, y: e.val_loss })) },
      ], { xLabel: 'logged epoch (both phases)', yLabel: 'loss' })}
      ${SI.lineChart([
        { label: 'validation accuracy', className: 'series-c', points: points.map((e) => ({ x: e.x, y: e.val_accuracy })) },
      ], { xLabel: 'logged epoch (both phases)', yLabel: 'accuracy', height: 180 })}
      <p class="muted small-print">${esc(history.note)} Marked points are those one-batch epochs.</p>
      <ul class="plain-list">${legacy.known_issues.map((i) => `<li>${esc(i)}</li>`).join('')}</ul>`;
  }

  function renderEvaluation({ metrics }) {
    const evaluation = metrics.evaluation;
    let html = '';
    if (evaluation && Object.keys(evaluation.models).length) {
      const history = metrics.training_history.resnet18;
      html += `<h4 class="section-title">Trained models ${provenance('measured')}</h4>
        <p class="muted">Evaluated ${esc(evaluation.created_utc)} on dataset ${esc(evaluation.dataset.dataset_fingerprint)} (${esc(evaluation.dataset.label_mode)}).</p>
        ${(evaluation.warnings || []).map((w) => `<p class="note">${esc(w)}</p>`).join('')}
        ${metricsTable(evaluation)}
        ${Object.entries(evaluation.models).map(([n, e]) => perModelDetails(n, e)).join('')}
        ${history ? `<h5>ResNet18 training history (best epoch ${history.best_epoch})</h5>${SI.lineChart([
          { label: 'train loss', className: 'series-a', points: history.epochs.map((e) => ({ x: e.epoch, y: e.train_loss })) },
          { label: 'validation loss', className: 'series-b', points: history.epochs.map((e) => ({ x: e.epoch, y: e.val_loss })) },
        ])}` : ''}
        ${metrics.plots.length ? `<div class="plot-grid">${metrics.plots.map((p) => `<img src="${esc(p)}" alt="Evaluation plot ${esc(p.split('/').pop())}" loading="lazy">`).join('')}</div>` : ''}`;
    } else {
      html += `<div class="empty-state">
        <h4>No evaluation results yet ${provenance('unavailable')}</h4>
        <p>The ResNet18, SVM and Random Forest models have not been trained, because no labelled dataset is available in this environment. Once you have one (see docs/DATASET.md), run:</p>
        <pre><code>python -m src.data.dataset --data-root data/hwxpi
python -m src.classical.train_classical
python -m src.deep_learning.train_resnet18
python -m src.evaluation.evaluate</code></pre></div>`;
    }
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
  });
  SI.onTab(show);
  show(location.hash.slice(1));
})();
