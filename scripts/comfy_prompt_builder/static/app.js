const $ = (selector) => document.querySelector(selector);
const backendSelect = $('#backend');
const ollamaOptions = $('#ollamaOptions');
const localOptions = $('#localOptions');
const ollamaModelInput = $('#ollamaModel');
const ollamaStatus = $('#ollamaStatus');
const localStatus = $('#localStatus');
const ggufInput = $('#ggufPath');
const generateBtn = $('#generate');
const resultBox = $('#result');
const parsedBox = $('#parsed');
const positiveBox = $('#positive');
const negativeBox = $('#negative');
const extrasBox = $('#extras');
const copyJsonBtn = $('#copyJson');
const copyPositiveBtn = $('#copyPositive');
const browseBtn = $('#browse');
const fileBrowser = $('#fileBrowser');
const closeBrowserBtn = $('#closeBrowser');
const entriesList = $('#entries');
const breadcrumbs = $('#breadcrumbs');
const entryTemplate = document.getElementById('entryTemplate');

const state = {
  backend: 'ollama',
  ollamaModels: [],
  browserPath: null,
  lastResponse: null,
};

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json();
}

async function loadOllamaModels() {
  try {
    const data = await fetchJSON('/api/ollama_models');
    state.ollamaModels = data.models || [];
    ollamaStatus.textContent = `Found ${state.ollamaModels.length} model(s).`;
    const datalist = $('#ollamaModels');
    datalist.innerHTML = '';
    state.ollamaModels.forEach((model) => {
      const option = document.createElement('option');
      option.value = model;
      datalist.appendChild(option);
    });
  } catch (error) {
    ollamaStatus.textContent = `Ollama unavailable: ${error.message}`;
    ollamaStatus.classList.add('error');
  }
}

async function loadLocalStatus() {
  try {
    const data = await fetchJSON('/api/local/status');
    if (!data.available) {
      localStatus.textContent = 'Install llama-cpp-python to enable local GGUF models.';
      localStatus.classList.add('error');
      return;
    }
    const details = [
      data.current_path ? `Current: ${data.current_path}` : 'No model loaded yet',
      `Context: ${data.context}`,
    ];
    if (data.threads) details.push(`Threads: ${data.threads}`);
    if (data.n_gpu_layers !== null && data.n_gpu_layers !== undefined) {
      details.push(`GPU layers: ${data.n_gpu_layers}`);
    }
    localStatus.textContent = details.join(' • ');
    localStatus.classList.remove('error');
  } catch (error) {
    localStatus.textContent = `Local runtime unavailable: ${error.message}`;
    localStatus.classList.add('error');
  }
}

function switchBackend(value) {
  state.backend = value;
  if (value === 'ollama') {
    ollamaOptions.classList.remove('hidden');
    localOptions.classList.add('hidden');
  } else {
    localOptions.classList.remove('hidden');
    ollamaOptions.classList.add('hidden');
    loadLocalStatus();
  }
}

function resetResult() {
  resultBox.textContent = 'Generating…';
  parsedBox.classList.add('hidden');
  positiveBox.textContent = '';
  negativeBox.textContent = '';
  extrasBox.textContent = '';
}

function renderResult(content) {
  state.lastResponse = content;
  resultBox.textContent = content;
  try {
    const parsed = JSON.parse(content);
    positiveBox.textContent = parsed.prompt || parsed.positive || '';
    negativeBox.textContent = parsed.negative || '';
    extrasBox.textContent = parsed.extras ? JSON.stringify(parsed.extras, null, 2) : '';
    parsedBox.classList.remove('hidden');
    state.lastJson = parsed;
  } catch (error) {
    parsedBox.classList.add('hidden');
    state.lastJson = null;
  }
}

async function generate() {
  const idea = $('#idea').value.trim();
  if (!idea) {
    alert('Please provide an idea.');
    return;
  }
  const payload = {
    backend: state.backend,
    idea,
    system: $('#system').value,
    examples: $('#examples').value,
    temperature: parseFloat($('#temperature').value),
    top_p: parseFloat($('#top_p').value),
    top_k: $('#top_k').value ? parseInt($('#top_k').value, 10) : null,
    repeat_penalty: $('#repeat_penalty').value ? parseFloat($('#repeat_penalty').value) : null,
    seed: $('#seed').value ? parseInt($('#seed').value, 10) : null,
    max_tokens: $('#max_tokens').value ? parseInt($('#max_tokens').value, 10) : null,
  };
  if (state.backend === 'ollama') {
    payload.model = ollamaModelInput.value.trim();
    if (!payload.model) {
      alert('Enter an Ollama model name.');
      return;
    }
  } else {
    payload.gguf_path = ggufInput.value.trim();
    if (!payload.gguf_path) {
      alert('Select a GGUF file.');
      return;
    }
  }
  resetResult();
  try {
    const res = await fetch('/api/generate_prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const message = await res.text();
      throw new Error(message);
    }
    const data = await res.json();
    renderResult(data.content);
  } catch (error) {
    resultBox.textContent = `Error: ${error.message}`;
  }
}

function copyToClipboard(text) {
  if (!text) return;
  navigator.clipboard.writeText(text).catch(() => {
    alert('Failed to copy to clipboard.');
  });
}

function copyJson() {
  if (state.lastResponse) {
    copyToClipboard(state.lastResponse);
  }
}

function copyPositive() {
  if (state.lastJson) {
    copyToClipboard(state.lastJson.prompt || state.lastJson.positive || '');
  } else if (state.lastResponse) {
    copyToClipboard(state.lastResponse);
  }
}

async function showBrowser(path = null) {
  fileBrowser.classList.remove('hidden');
  await loadDirectory(path);
}

function hideBrowser() {
  fileBrowser.classList.add('hidden');
}

function renderBreadcrumbs(path) {
  if (!path) {
    breadcrumbs.textContent = 'Root';
    return;
  }
  breadcrumbs.textContent = path;
}

function createEntry(item, type) {
  const element = entryTemplate.content.firstElementChild.cloneNode(true);
  const button = element.querySelector('button');
  button.textContent = item.name;
  if (type === 'dir') {
    const span = document.createElement('span');
    span.textContent = 'Open';
    button.appendChild(span);
    button.addEventListener('click', () => loadDirectory(item.path));
  } else {
    const span = document.createElement('span');
    span.textContent = 'Use';
    button.appendChild(span);
    button.addEventListener('click', () => {
      ggufInput.value = item.path;
      hideBrowser();
      loadLocalStatus();
    });
  }
  return element;
}

async function loadDirectory(path = null) {
  entriesList.innerHTML = '';
  try {
    const url = new URL('/api/filetree', window.location.origin);
    if (path) {
      url.searchParams.set('path', path);
    }
    const data = await fetchJSON(url);
    state.browserPath = data.path;
    renderBreadcrumbs(data.path);
    if (data.parent) {
      const parentEntry = createEntry({ name: '…', path: data.parent }, 'dir');
      entriesList.appendChild(parentEntry);
    }
    data.directories.forEach((dir) => {
      const dirEntry = createEntry(dir, 'dir');
      entriesList.appendChild(dirEntry);
    });
    if (data.files.length === 0 && data.directories.length === 0) {
      const empty = document.createElement('li');
      empty.textContent = 'No entries';
      entriesList.appendChild(empty);
    }
    data.files.forEach((file) => {
      const fileEntry = createEntry(file, 'file');
      entriesList.appendChild(fileEntry);
    });
  } catch (error) {
    const item = document.createElement('li');
    item.textContent = `Unable to list entries: ${error.message}`;
    entriesList.appendChild(item);
  }
}

backendSelect.addEventListener('change', (event) => {
  switchBackend(event.target.value);
});

generateBtn.addEventListener('click', generate);
copyJsonBtn.addEventListener('click', copyJson);
copyPositiveBtn.addEventListener('click', copyPositive);
browseBtn.addEventListener('click', () => showBrowser(state.browserPath));
closeBrowserBtn.addEventListener('click', hideBrowser);
fileBrowser.addEventListener('click', (event) => {
  if (event.target === fileBrowser) {
    hideBrowser();
  }
});

window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !fileBrowser.classList.contains('hidden')) {
    hideBrowser();
  }
});

switchBackend(backendSelect.value);
loadOllamaModels();
