const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const resultPanel = document.getElementById('resultPanel');
const resultModel = document.getElementById('resultModel');
const resultNotes = document.getElementById('resultNotes');
const predictedTrait = document.getElementById('predictedTrait');
const confidenceBadge = document.getElementById('confidenceBadge');
const confidenceLabel = document.getElementById('confidenceLabel');
const confidenceValue = document.getElementById('confidenceValue');
const probabilityGrid = document.getElementById('probabilityGrid');
const comparisonPanel = document.getElementById('comparisonPanel');
const comparisonTable = document.getElementById('comparisonTable');
const traitDetails = document.getElementById('traitDetails');
const traitDetailsContent = document.getElementById('traitDetailsContent');
const selectedFileLabel = document.getElementById('selectedFileLabel');
const selectedPreview = document.getElementById('selectedPreview');
const selectedPreviewWrap = document.getElementById('selectedPreviewWrap');
const fileMeta = document.getElementById('fileMeta');
const fileMessage = document.getElementById('fileMessage');
const analyzeBtn = document.getElementById('analyzeBtn');
const replaceBtn = document.getElementById('replaceBtn');
const exportBtn = document.getElementById('exportBtn');
const historyList = document.getElementById('historyList');
const historyToggle = document.getElementById('historyToggle');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');

const MAX_FILE_SIZE = 16 * 1024 * 1024;
const MIN_SIDE = 128;
const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tif', 'tiff', 'gif'];
const IDLE_MESSAGE = 'Choose or drop a handwriting image to enable analysis.';
const HISTORY_KEY = 'scriptinsight-history';
const HISTORY_ENABLED_KEY = 'scriptinsight-history-enabled';
const HISTORY_LIMIT = 20;

// The selected file lives here rather than only in fileInput.files, so a canceled
// file picker (which some browsers report as an empty selection) cannot erase it.
let selectedFile = null;
let selectedDims = null;
let isAnalyzing = false;
let lastReport = null;
let selectionToken = 0;

const TRAIT_DETAILS = {
  Openness: {
    emoji: '🌟',
    definition: 'A tendency to be curious, imaginative, and open to new ideas and experiences.',
    characteristics: [
      'Creative and imaginative thinking',
      'Interested in novelty and variety',
      'Curious about new perspectives',
      'Enjoys intellectual exploration',
      'Appreciates artistic and aesthetic experiences'
    ],
    strengths: [
      'Strong creativity and innovation',
      'Comfort with complex ideas',
      'Adaptable to change',
      'Excellent problem-solving perspective',
      'Broad interests'
    ],
    weaknesses: [
      'May become distracted by too many ideas',
      'Can be overly idealistic',
      'May resist structure or routine',
      'Can struggle with decision fatigue'
    ]
  },
  Conscientiousness: {
    emoji: '✅',
    definition: 'A tendency to be organized, reliable, disciplined, and goal-focused.',
    characteristics: [
      'Highly organized and structured',
      'Responsible and dependable',
      'Goal-oriented and persistent',
      'Detail-conscious and careful',
      'Strong self-discipline'
    ],
    strengths: [
      'Excellent planning skills',
      'Strong work ethic',
      'High reliability',
      'Attention to detail',
      'Consistent performance'
    ],
    weaknesses: [
      'Can become overly rigid',
      'May be perfectionistic',
      'Can struggle to relax',
      'May be uncomfortable with spontaneity'
    ]
  },
  Extraversion: {
    emoji: '🎉',
    definition: 'A tendency to seek energy from social interaction, stimulation, and engagement.',
    characteristics: [
      'Outgoing and socially confident',
      'Enjoys group interaction',
      'Energetic and enthusiastic',
      'Talkative and engaging',
      'Motivated by social stimulation'
    ],
    strengths: [
      'Strong communication skills',
      'High social energy',
      'Good team collaboration',
      'Confident leadership potential',
      'Positive influence on groups'
    ],
    weaknesses: [
      'May become overstimulated',
      'Can interrupt or dominate conversations',
      'May seek constant attention',
      'Can become restless when alone'
    ]
  },
  Agreeableness: {
    emoji: '💚',
    definition: 'A tendency to be compassionate, cooperative, and considerate of others.',
    characteristics: [
      'Empathetic and understanding',
      'Friendly and cooperative',
      'Trusting and supportive',
      'Values harmony and connection',
      'Kind in interpersonal interactions'
    ],
    strengths: [
      'Strong interpersonal skills',
      'Trust-building and diplomacy',
      'Conflict resolution ability',
      'Supportive team member',
      'Warm and approachable nature'
    ],
    weaknesses: [
      'May avoid conflict too much',
      'Can be overly accommodating',
      'May internalize others\' stress',
      'May struggle with firm boundaries'
    ]
  },
  Neuroticism: {
    emoji: '⚡',
    definition: 'A tendency toward emotional sensitivity, stress reactivity, and negative mood fluctuations.',
    characteristics: [
      'Emotionally responsive',
      'Sensitive to stress and uncertainty',
      'Highly self-aware',
      'Can feel strong emotions deeply',
      'Tends to reflect on worries and concerns'
    ],
    strengths: [
      'High emotional awareness',
      'Intuitive understanding of feelings',
      'Strong vigilance to risk',
      'Reflective and insightful',
      'High empathy toward emotional nuance'
    ],
    weaknesses: [
      'More prone to anxiety or worry',
      'May experience emotional overwhelm',
      'Can be sensitive to criticism',
      'May struggle with stress management'
    ]
  }
};

// tone: 'error' | 'success' | '' (neutral)
function setFileMessage(message, tone = '') {
  fileMessage.textContent = message;
  fileMessage.classList.toggle('error', tone === 'error');
  fileMessage.classList.toggle('success', tone === 'success');
}

function updateAnalyzeButtonState() {
  const canAnalyze = !!selectedFile && !isAnalyzing;
  analyzeBtn.disabled = !canAnalyze;
  analyzeBtn.setAttribute('aria-busy', String(isAnalyzing));
}

function clearPreview() {
  if (selectedPreview.src.startsWith('blob:')) {
    URL.revokeObjectURL(selectedPreview.src);
  }
  selectedPreview.removeAttribute('src');
  selectedPreviewWrap.classList.add('hidden');
}

function validateSelectedFile(file) {
  if (!file) {
    return 'Please choose an image file first.';
  }

  const extension = file.name.includes('.') ? file.name.split('.').pop().toLowerCase() : '';
  if (!ALLOWED_EXTENSIONS.includes(extension)) {
    return 'Unsupported image type. Please choose JPG, PNG or WEBP (BMP, TIFF and GIF also work).';
  }

  if (file.size === 0) {
    return 'This image file is empty. Please choose another image.';
  }

  if (file.size > MAX_FILE_SIZE) {
    return 'Image is too large. Please select an image smaller than 16MB.';
  }

  return '';
}

// Resolves to {width, height}, or null when this browser cannot decode the format (e.g. TIFF).
function readDimensions(url) {
  return new Promise((resolve) => {
    const img = new Image();
    const timer = setTimeout(() => resolve(null), 8000);
    img.onload = () => { clearTimeout(timer); resolve({ width: img.naturalWidth, height: img.naturalHeight }); };
    img.onerror = () => { clearTimeout(timer); resolve(null); };
    img.src = url;
  });
}

async function selectFile(file) {
  if (!file) {
    // Picker was canceled: keep whatever was selected before.
    if (!selectedFile) {
      setFileMessage(IDLE_MESSAGE);
    }
    return;
  }

  const reject = (message) => {
    // Keep a previously selected valid image; just report the problem with the new one.
    setFileMessage(selectedFile ? `${message} Still using ${selectedFile.name}.` : message, 'error');
  };

  const validationError = validateSelectedFile(file);
  if (validationError) {
    reject(validationError);
    return;
  }

  const token = ++selectionToken;
  const url = URL.createObjectURL(file);
  const dims = await readDimensions(url);
  if (token !== selectionToken) {
    URL.revokeObjectURL(url);  // a newer selection replaced this one while it was loading
    return;
  }
  if (dims && (dims.width < MIN_SIDE || dims.height < MIN_SIDE)) {
    URL.revokeObjectURL(url);
    reject(`Image is too small (${dims.width}×${dims.height}). Minimum size is ${MIN_SIDE}×${MIN_SIDE} pixels.`);
    return;
  }

  selectedFile = file;
  selectedDims = dims;
  clearPreview();
  if (dims) {
    selectedPreview.src = url;
    selectedPreviewWrap.classList.remove('hidden');
  } else {
    URL.revokeObjectURL(url);
  }
  selectedFileLabel.textContent = file.name;
  fileMeta.textContent = [
    SI.formatBytes(file.size),
    dims ? `${dims.width}×${dims.height} px` : 'Preview not supported by this browser; the server will check the image'
  ].join(' · ');
  fileMeta.classList.remove('hidden');
  replaceBtn.classList.remove('hidden');
  setFileMessage('Image ready for analysis.', 'success');
  updateAnalyzeButtonState();
}

function handleFileSelection(event) {
  const file = event.target.files && event.target.files[0];
  // Reset the input so choosing the same file again still fires 'change'.
  fileInput.value = '';
  selectFile(file);
}

function barRow(label, value, caption) {
  const percent = Math.max(0, Math.min(100, value * 100));
  return `
    <div class="probability-item">
      <div class="probability-label-row">
        <span>${SI.escapeHtml(label)}</span>
        <strong>${percent.toFixed(1)}%</strong>
      </div>
      <div class="bar-track" role="progressbar" aria-label="${SI.escapeHtml(label)}"
           aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent.toFixed(1)}">
        <div class="bar-fill" style="width: ${percent.toFixed(1)}%"></div>
      </div>
      ${caption ? `<p class="bar-caption">${SI.escapeHtml(caption)}</p>` : ''}
    </div>`;
}

function renderResult(data) {
  const prediction = data.prediction;
  const primary = data.model_predictions.find((m) => m.model === prediction.model) || {};
  resultModel.textContent = `Primary model: ${prediction.display_name}`;

  if (prediction.task === 'multiclass') {
    predictedTrait.textContent = prediction.predicted_personality;
    confidenceBadge.classList.remove('hidden');
    confidenceLabel.textContent = 'Top category score';
    confidenceValue.textContent = SI.pct(prediction.confidence);
    probabilityGrid.innerHTML = SI.TRAITS
      .filter((trait) => trait in prediction.probabilities)
      .map((trait) => barRow(trait, prediction.probabilities[trait]))
      .join('');
    displayTraitDetails(prediction.predicted_personality);
  } else {
    predictedTrait.textContent = 'Five independent trait estimates';
    confidenceBadge.classList.add('hidden');
    probabilityGrid.innerHTML = SI.TRAITS
      .filter((trait) => trait in prediction.traits)
      .map((trait) => {
        const t = prediction.traits[trait];
        return barRow(trait, t.probability, `Estimated ${t.level} pole (score of the high pole; threshold 50%)`);
      })
      .join('');
    traitDetails.classList.add('hidden');
  }

  const notes = [...(data.warnings || [])];
  if (primary.score_note) notes.push(`Scores: ${primary.score_note}`);
  resultNotes.innerHTML = notes.map((n) => `<p class="note">${SI.escapeHtml(n)}</p>`).join('');
  resultPanel.classList.remove('hidden');
}

function scoreSummary(entry) {
  if (entry.task === 'multilabel') {
    return SI.TRAITS.filter((t) => t in entry.scores)
      .map((t) => `${t}: ${SI.pct(entry.scores[t])} (${entry.levels[t]})`).join('<br>');
  }
  return SI.TRAITS.filter((t) => t in entry.scores)
    .map((t) => `${t === entry.predicted_label ? '<strong>' : ''}${t}: ${SI.pct(entry.scores[t])}${t === entry.predicted_label ? '</strong>' : ''}`)
    .join('<br>');
}

function renderComparison(models) {
  comparisonTable.innerHTML = `
    <thead><tr><th scope="col">Model</th><th scope="col">Output for this image</th><th scope="col">Scores</th><th scope="col">Score type</th><th scope="col">Time</th></tr></thead>
    <tbody>${models.map((m) => {
      if (m.status !== 'ok') {
        return `<tr class="row-unavailable"><th scope="row">${SI.escapeHtml(m.display_name)}</th>
          <td colspan="4"><span class="chip chip-muted">${m.status === 'error' ? 'Error' : 'Unavailable'}</span> ${SI.escapeHtml(m.message || '')}</td></tr>`;
      }
      const output = m.task === 'multilabel' ? 'Per-trait high/low estimates' : `Category: <strong>${SI.escapeHtml(m.predicted_label)}</strong>`;
      return `<tr><th scope="row">${SI.escapeHtml(m.display_name)}</th><td>${output}</td><td>${scoreSummary(m)}</td>
        <td title="${SI.escapeHtml(m.score_note || '')}">${SI.escapeHtml((m.score_type || '').replace(/_/g, ' '))}</td>
        <td>${m.latency_ms} ms</td></tr>`;
    }).join('')}</tbody>`;
  comparisonPanel.classList.remove('hidden');
}

// ----- History (sessionStorage only; never stores the image) -----
function historyEnabled() {
  try { return sessionStorage.getItem(HISTORY_ENABLED_KEY) !== 'false'; } catch (_) { return false; }
}

function readHistory() {
  try { return JSON.parse(sessionStorage.getItem(HISTORY_KEY) || '[]'); } catch (_) { return []; }
}

function writeHistory(items) {
  try { sessionStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_LIMIT))); } catch (_) { /* storage blocked */ }
}

function renderHistory() {
  const items = readHistory();
  if (!items.length) {
    historyList.innerHTML = `<p class="empty-note">${historyEnabled() ? 'No analyses in this tab yet.' : 'History is turned off.'}</p>`;
    return;
  }
  historyList.innerHTML = `<ul class="history-items">${items.map((item, index) => `
    <li>
      <div><strong>${SI.escapeHtml(item.file.name)}</strong>
        <span class="muted">· ${SI.escapeHtml(new Date(item.time).toLocaleString())}</span></div>
      <div class="muted">${SI.escapeHtml(item.data.prediction.display_name)}:
        ${SI.escapeHtml(item.data.prediction.task === 'multiclass'
          ? `${item.data.prediction.predicted_personality} (${SI.pct(item.data.prediction.confidence)})`
          : 'per-trait estimates')}</div>
      <button type="button" class="secondary-btn small" data-export-index="${index}">Download report</button>
    </li>`).join('')}</ul>`;
}

historyList.addEventListener('click', (event) => {
  const button = event.target.closest('[data-export-index]');
  if (button) downloadReport(readHistory()[Number(button.dataset.exportIndex)]);
});

historyToggle.checked = historyEnabled();
historyToggle.addEventListener('change', () => {
  try { sessionStorage.setItem(HISTORY_ENABLED_KEY, String(historyToggle.checked)); } catch (_) { /* storage blocked */ }
  if (!historyToggle.checked) writeHistory([]);
  renderHistory();
});

clearHistoryBtn.addEventListener('click', () => {
  writeHistory([]);
  renderHistory();
});

// ----- Report export (generated from the actual API response) -----
function downloadReport(report) {
  if (!report) return;
  const { data, file, time } = report;
  const rows = data.model_predictions.map((m) => `<tr><td>${SI.escapeHtml(m.display_name)}</td><td>${
    m.status === 'ok' ? scoreSummary(m) : `${SI.escapeHtml(m.status)}: ${SI.escapeHtml(m.message || '')}`}</td><td>${
    SI.escapeHtml(m.score_type || '')}</td></tr>`).join('');
  const html = `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>ScriptInsight report</title>
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:32px auto;padding:0 16px;color:#111}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}
.warn{background:#fff7e6;border:1px solid #f0c36d;padding:10px;border-radius:6px}</style></head><body>
<h1>ScriptInsight analysis report</h1>
<p class="warn"><strong>Please note:</strong> ${SI.escapeHtml(data.disclaimer)}</p>
<p><strong>Image:</strong> ${SI.escapeHtml(file.name)} (${SI.escapeHtml(SI.formatBytes(file.size))}, ${data.image.width}×${data.image.height}, ${SI.escapeHtml(data.image.format)})<br>
<strong>Analysed:</strong> ${SI.escapeHtml(new Date(time).toLocaleString())}<br>
<strong>App version:</strong> ${SI.escapeHtml(data.model_metadata.app_version)} · <strong>Primary model:</strong> ${SI.escapeHtml(data.prediction.display_name)}</p>
<h2>Model outputs</h2><table><thead><tr><th>Model</th><th>Scores</th><th>Score type</th></tr></thead><tbody>${rows}</tbody></table>
${(data.warnings || []).length ? `<h2>Warnings</h2><ul>${data.warnings.map((w) => `<li>${SI.escapeHtml(w)}</li>`).join('')}</ul>` : ''}
<p>Scores are model outputs, not a psychological assessment. See the application's Evaluation tab for measured accuracy.</p>
</body></html>`;
  const blob = new Blob([html], { type: 'text/html' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `scriptinsight-report-${new Date(time).toISOString().replace(/[:.]/g, '-')}.html`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

exportBtn.addEventListener('click', () => downloadReport(lastReport));

// ----- Upload interactions -----
// The <label for="fileInput"> already opens the file picker natively; no extra click handler.
fileInput.addEventListener('change', handleFileSelection);

// Keyboard access: the label is focusable via tabindex, so open the picker on Enter/Space.
uploadBox.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
});

replaceBtn.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover'].forEach((type) => uploadBox.addEventListener(type, (event) => {
  event.preventDefault();
  uploadBox.classList.add('dragging');
}));
['dragleave', 'dragend', 'drop'].forEach((type) => uploadBox.addEventListener(type, () => {
  uploadBox.classList.remove('dragging');
}));
uploadBox.addEventListener('drop', (event) => {
  event.preventDefault();
  const files = event.dataTransfer && event.dataTransfer.files;
  if (files && files.length > 1) setFileMessage('Please drop one image at a time.', 'error');
  else selectFile(files && files[0]);
});
// Dropping a file elsewhere on the page must not navigate away from the app.
['dragover', 'drop'].forEach((type) => window.addEventListener(type, (event) => event.preventDefault()));

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (isAnalyzing) return;

  const file = selectedFile;
  const validationError = validateSelectedFile(file);

  if (validationError) {
    setFileMessage(validationError, 'error');
    updateAnalyzeButtonState();
    return;
  }

  const formData = new FormData();
  formData.append('file', file, file.name);

  isAnalyzing = true;
  updateAnalyzeButtonState();
  analyzeBtn.textContent = 'Analyzing...';
  setFileMessage(`Uploading and analyzing ${file.name}...`);

  try {
    let response;
    try {
      response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });
    } catch (networkError) {
      throw new Error('Could not reach the server. Make sure the app is running and try again.');
    }

    if (!response.ok) {
      throw new Error(await SI.errorMessage(response));
    }

    const data = await response.json();
    if (!data.success || !data.prediction) {
      throw new Error(data.error || 'Prediction failed');
    }

    renderResult(data);
    renderComparison(data.model_predictions);
    lastReport = { data, file: { name: file.name, size: file.size }, time: new Date().toISOString() };
    if (historyEnabled()) {
      writeHistory([lastReport, ...readHistory()]);
      renderHistory();
    }
    setFileMessage('Analysis complete.', 'success');
  } catch (error) {
    setFileMessage(error.message || 'Something went wrong while analyzing the image.', 'error');
    console.warn('Upload/analysis error:', error);
  } finally {
    isAnalyzing = false;
    analyzeBtn.textContent = 'Analyze handwriting';
    updateAnalyzeButtonState();
  }
});

function displayTraitDetails(traitName) {
  const trait = TRAIT_DETAILS[traitName];
  if (!trait) {
    traitDetails.classList.add('hidden');
    return;
  }

  traitDetailsContent.innerHTML = `
    <div class="trait-content">
      <div class="trait-card">
        <div class="trait-detail-header">
          <span class="trait-emoji">${trait.emoji}</span>
          <h4>${traitName}</h4>
        </div>
        <p>${trait.definition}</p>
      </div>

      <div class="trait-card">
        <h4>Key characteristics</h4>
        <ul class="characteristics-list">
          ${trait.characteristics.map((item) => `<li><span class="checkmark">✓</span> ${item}</li>`).join('')}
        </ul>
      </div>

      <div class="trait-card">
        <h4>Commonly associated strengths</h4>
        <ul class="strength-list">
          ${trait.strengths.map((item) => `<li><span class="checkmark">✓</span> ${item}</li>`).join('')}
        </ul>
      </div>

      <div class="trait-card">
        <h4>Commonly associated challenges</h4>
        <ul class="weakness-list">
          ${trait.weaknesses.map((item) => `<li><span class="warning">⚠</span> ${item}</li>`).join('')}
        </ul>
      </div>
    </div>
  `;
  traitDetails.classList.remove('hidden');
}

SI.initTheme(document.getElementById('themeToggle'), document.getElementById('themeIcon'));
SI.initTabs();
setFileMessage(IDLE_MESSAGE);
updateAnalyzeButtonState();
renderHistory();
