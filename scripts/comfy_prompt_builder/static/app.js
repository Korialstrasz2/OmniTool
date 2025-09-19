const statusMessage = document.getElementById('connectionStatus');
const retryButton = document.getElementById('retryConnection');
const ideaInput = document.getElementById('ideaInput');
const sendButton = document.getElementById('sendIdea');
const responseBox = document.getElementById('responseBox');

let koboldUrl = null;

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle('status__message--error', isError);
}

async function checkConnection() {
  setStatus('Checking Kobold connection…');
  try {
    const res = await fetch('/api/kobold/status');
    if (!res.ok) {
      throw new Error(await res.text());
    }
    const data = await res.json();
    koboldUrl = data.url || null;
    if (data.online) {
      const label = data.model ? `Connected to ${data.model}` : 'Kobold connection established';
      setStatus(label, false);
    } else if (data.error) {
      setStatus(`Unable to reach Kobold: ${data.error}`, true);
    } else {
      setStatus('Kobold instance not detected yet.', true);
    }
  } catch (error) {
    setStatus(`Connection check failed: ${error.message}`, true);
  }
}

async function sendIdea() {
  const idea = ideaInput.value.trim();
  if (!idea) {
    setStatus('Please enter an idea before sending.', true);
    ideaInput.focus();
    return;
  }

  responseBox.textContent = 'Waiting for Kobold…';
  try {
    const res = await fetch('/api/generate_prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ idea, kobold_url: koboldUrl }),
    });
    if (!res.ok) {
      const message = await res.text();
      throw new Error(message || 'Request failed');
    }
    const data = await res.json();
    responseBox.textContent = data.content || 'Kobold returned no text.';
    if (data.model) {
      setStatus(`Connected to ${data.model}`, false);
    }
  } catch (error) {
    responseBox.textContent = `Error: ${error.message}`;
    setStatus('Prompt request failed. Retry after checking Kobold.', true);
  }
}

retryButton.addEventListener('click', checkConnection);
sendButton.addEventListener('click', sendIdea);
ideaInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
    sendIdea();
  }
});

checkConnection();
