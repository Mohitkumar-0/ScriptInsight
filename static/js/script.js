const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.getElementById('themeIcon');
const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const resultPanel = document.getElementById('resultPanel');
const predictedTrait = document.getElementById('predictedTrait');
const confidenceValue = document.getElementById('confidenceValue');
const probabilityGrid = document.getElementById('probabilityGrid');
const traitDetails = document.getElementById('traitDetails');
const traitDetailsContent = document.getElementById('traitDetailsContent');
const selectedFileLabel = document.getElementById('selectedFileLabel');
const selectedPreview = document.getElementById('selectedPreview');
const selectedPreviewWrap = document.getElementById('selectedPreviewWrap');
const fileMessage = document.getElementById('fileMessage');
const analyzeBtn = document.getElementById('analyzeBtn');

const MAX_FILE_SIZE = 16 * 1024 * 1024;
const ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tif', 'tiff'];
const IDLE_MESSAGE = 'Choose a handwriting image to enable analysis.';

// The selected file lives here rather than only in fileInput.files, so a canceled
// file picker (which some browsers report as an empty selection) cannot erase it.
let selectedFile = null;
let isAnalyzing = false;

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

function showPreview(file) {
  clearPreview();
  selectedPreview.src = URL.createObjectURL(file);
  selectedPreviewWrap.classList.remove('hidden');
}

function validateSelectedFile(file) {
  if (!file) {
    return 'Please choose an image file first.';
  }

  const extension = file.name.includes('.') ? file.name.split('.').pop().toLowerCase() : '';
  if (!ALLOWED_EXTENSIONS.includes(extension)) {
    return 'Unsupported image type. Please choose PNG, JPG, JPEG, GIF, BMP, or TIFF.';
  }

  if (file.size === 0) {
    return 'This image file is empty. Please choose another image.';
  }

  if (file.size > MAX_FILE_SIZE) {
    return 'Image is too large. Please select an image smaller than 16MB.';
  }

  return '';
}

function handleFileSelection(event) {
  const file = event.target.files && event.target.files[0];

  // Reset the input so choosing the same file again still fires 'change'.
  fileInput.value = '';

  if (!file) {
    // Picker was canceled: keep whatever was selected before.
    if (!selectedFile) {
      setFileMessage(IDLE_MESSAGE);
    }
    return;
  }

  const validationError = validateSelectedFile(file);
  if (validationError) {
    // Keep a previously selected valid image; just report the problem with the new one.
    setFileMessage(
      selectedFile ? `${validationError} Still using ${selectedFile.name}.` : validationError,
      'error'
    );
    return;
  }

  selectedFile = file;
  selectedFileLabel.textContent = file.name;
  showPreview(file);
  setFileMessage('Image ready for analysis.', 'success');
  updateAnalyzeButtonState();
}

async function readErrorMessage(response) {
  try {
    const data = await response.json();
    if (data && data.error) return data.error;
  } catch (_) {
    // Non-JSON error body (e.g. a proxy or HTML error page).
  }
  if (response.status === 413) return 'Image is too large. Please select an image smaller than 16MB.';
  return `The server could not analyze the image (HTTP ${response.status}). Please try again.`;
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);
  themeIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
}

function initTheme() {
  const saved = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const initial = saved || (prefersDark ? 'dark' : 'light');
  applyTheme(initial);
}

themeToggle.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  applyTheme(current);
});

// The <label for="fileInput"> already opens the file picker natively. Calling
// fileInput.click() here as well opened the picker twice, and dismissing the
// second dialog cleared the selection — so no extra click handler is attached.
fileInput.addEventListener('change', handleFileSelection);

// Keyboard access: the label is focusable via tabindex, so open the picker on Enter/Space.
uploadBox.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
});

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
      throw new Error(await readErrorMessage(response));
    }

    const data = await response.json();
    if (!data.success || !data.prediction) {
      throw new Error(data.error || 'Prediction failed');
    }

    const prediction = data.prediction;
    const probabilities = prediction.probabilities;

    predictedTrait.textContent = prediction.predicted_personality;
    confidenceValue.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;

    const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    probabilityGrid.innerHTML = entries.map(([trait, value]) => `
      <div class="probability-item">
        <div class="probability-label-row">
          <span>${trait}</span>
          <strong>${(value * 100).toFixed(1)}%</strong>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width: ${(value * 100).toFixed(1)}%"></div>
        </div>
      </div>
    `).join('');

    displayTraitDetails(prediction.predicted_personality);
    resultPanel.classList.remove('hidden');
    traitDetails.classList.remove('hidden');
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
    traitDetailsContent.innerHTML = '<p>No details available.</p>';
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
        <h4>Strengths</h4>
        <ul class="strength-list">
          ${trait.strengths.map((item) => `<li><span class="checkmark">✓</span> ${item}</li>`).join('')}
        </ul>
      </div>

      <div class="trait-card">
        <h4>Potential challenges</h4>
        <ul class="weakness-list">
          ${trait.weaknesses.map((item) => `<li><span class="warning">⚠</span> ${item}</li>`).join('')}
        </ul>
      </div>
    </div>
  `;
}

initTheme();
setFileMessage(IDLE_MESSAGE);
updateAnalyzeButtonState();
