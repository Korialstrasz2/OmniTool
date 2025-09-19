const $ = (selector) => document.querySelector(selector);
const backendSelect = $('#backend');
const koboldOptions = $('#koboldOptions');
const localOptions = $('#localOptions');
const koboldUrlInput = $('#koboldUrl');
const koboldStatus = $('#koboldStatus');
const localStatus = $('#localStatus');
const ggufInput = $('#ggufPath');
const generateBtn = $('#generate');
const resultBox = $('#result');
const copyPromptBtn = $('#copyPrompt');
const browseBtn = $('#browse');
const fileBrowser = $('#fileBrowser');
const closeBrowserBtn = $('#closeBrowser');
const entriesList = $('#entries');
const breadcrumbs = $('#breadcrumbs');
const entryTemplate = document.getElementById('entryTemplate');

const state = {
  backend: 'koboldcpp',
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

async function loadKoboldStatus() {
  const base = koboldUrlInput.value.trim();
  const query = base ? `?url=${encodeURIComponent(base)}` : '';
  try {
    const data = await fetchJSON(`/api/kobold/status${query}`);
    if (data.url && !base) {
      koboldUrlInput.value = data.url;
    }
    if (data.online) {
      const label = data.model ? `Connected to ${data.model}` : 'Connected to KoboldCpp';
      koboldStatus.textContent = label;
      koboldStatus.classList.remove('error');
    } else if (data.error) {
      koboldStatus.textContent = `Unable to reach KoboldCpp: ${data.error}`;
      koboldStatus.classList.add('error');
    } else {
      koboldStatus.textContent = 'KoboldCpp instance not detected yet.';
      koboldStatus.classList.add('error');
    }
  } catch (error) {
    koboldStatus.textContent = `KoboldCpp unavailable: ${error.message}`;
    koboldStatus.classList.add('error');
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
  if (value === 'koboldcpp') {
    koboldOptions.classList.remove('hidden');
    localOptions.classList.add('hidden');
    loadKoboldStatus();
  } else {
    localOptions.classList.remove('hidden');
    koboldOptions.classList.add('hidden');
    loadLocalStatus();
  }
}

function resetResult() {
  resultBox.textContent = 'Generating…';
}

function renderResult(content) {
  state.lastResponse = content;
  resultBox.textContent = content;
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
  if (state.backend === 'koboldcpp') {
    const url = koboldUrlInput.value.trim();
    if (url) {
      payload.kobold_url = url;
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

function copyPrompt() {
  if (state.lastResponse) {
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
copyPromptBtn.addEventListener('click', copyPrompt);
browseBtn.addEventListener('click', () => showBrowser(state.browserPath));
closeBrowserBtn.addEventListener('click', hideBrowser);
fileBrowser.addEventListener('click', (event) => {
  if (event.target === fileBrowser) {
    hideBrowser();
  }
});

koboldUrlInput.addEventListener('change', loadKoboldStatus);
koboldUrlInput.addEventListener('blur', loadKoboldStatus);

window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !fileBrowser.classList.contains('hidden')) {
    hideBrowser();
  }
});

switchBackend(backendSelect.value);
loadKoboldStatus();
