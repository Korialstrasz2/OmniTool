'use strict';
(() => {
  const app = document.getElementById('workspace-app');
  if (!app) return;
  const mode = app.dataset.mode;
  const api = window.OmniAPI;
  const results = document.getElementById('results');
  const message = document.getElementById('app-message');
  const instructions = document.getElementById('instructions');
  const unlockDialog = document.getElementById('unlock-dialog');
  const bundleDialog = document.getElementById('bundle-dialog');
  let page = 1, pages = 1, generation = 0, activeUnlock = null, activeBundle = null, privateGeneration = 0;
  let favorites = [];
  try { favorites = JSON.parse(localStorage.getItem('omnitool:favorites') || '[]'); } catch (_) { /* unavailable storage */ }
  if (!Array.isArray(favorites)) favorites = [];
  let favoritesOnly = new URLSearchParams(location.search).has('favorites');
  const element = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  const button = (text, action, cls = 'secondary') => {
    const b = element('button', text, cls); b.type = 'button'; b.addEventListener('click', action); return b;
  };
  const report = error => { message.textContent = error.message || String(error); message.hidden = false; results.setAttribute('aria-busy', 'false'); };
  const clearMessage = () => { message.hidden = true; message.textContent = ''; };
  function empty(title, text, action) {
    const box = element('section', undefined, 'empty-state');
    box.append(element('h2', title), element('p', text));
    if (action) box.append(action);
    results.append(box);
  }
  document.querySelector(`[data-nav="${mode}"]`)?.classList.add('active');
  document.getElementById('add-tool')?.addEventListener('click', () => instructions.showModal());
  document.querySelectorAll('[data-close]').forEach(b => b.addEventListener('click', () => document.getElementById(b.dataset.close).close()));
  unlockDialog.addEventListener('cancel', () => { document.getElementById('vault-passphrase').value = ''; });
  unlockDialog.addEventListener('close', () => {
    document.getElementById('vault-passphrase').value = '';
    document.getElementById('unlock-error').textContent = '';
    activeUnlock = null;
  });
  function clearPrivate() {
    privateGeneration++;
    activeBundle = null;
    document.getElementById('bundle-name').textContent = '';
    document.getElementById('bundle-tools').replaceChildren();
    document.getElementById('bundle-files').replaceChildren();
    document.getElementById('bundle-pages').replaceChildren();
  }
  bundleDialog.addEventListener('close', clearPrivate);
  function hidePrivate() { if (bundleDialog.open) bundleDialog.close(); clearPrivate(); }

  function toolCard(tool) {
    const card = element('article', undefined, 'tool-card');
    const top = element('div', undefined, 'card-top');
    top.append(element('span', tool.name.slice(0, 2).toUpperCase(), 'tool-icon'));
    const badge = element('span', tool.availability.replaceAll('-', ' '), `badge ${tool.availability}`);
    top.append(badge);
    const star = button(favorites.includes(tool.id) ? '★' : '☆', () => {
      favorites = favorites.includes(tool.id) ? favorites.filter(x => x !== tool.id) : [...favorites, tool.id];
      try { localStorage.setItem('omnitool:favorites', JSON.stringify(favorites)); } catch (_) { /* in-memory fallback */ }
      loadLibrary();
    }, 'favorite-button');
    star.setAttribute('aria-label', `Favorite ${tool.name}`);
    star.setAttribute('aria-pressed', String(favorites.includes(tool.id)));
    top.append(star);
    const footer = element('footer');
    const open = element('a', tool.availability === 'ready' ? 'Open tool →' : 'Review setup →');
    open.href = `/tool/${encodeURIComponent(tool.id)}`;
    footer.append(element('span', tool.folder, 'folder-tag'), open);
    card.append(top, element('h2', tool.name), element('p', tool.description), footer);
    return card;
  }
  async function loadLibrary() {
    const current = ++generation;
    results.setAttribute('aria-busy', 'true');
    try {
      const folder = document.getElementById('folder');
      const data = await api('/api/catalog', {query: document.getElementById('search').value, folder: folder.value,
        status: document.getElementById('status').value, page, size: 24, pinned: favorites, favorites_only: favoritesOnly});
      if (current !== generation) return;
      clearMessage();
      const selected = folder.value;
      folder.replaceChildren(new Option('All folders', ''), ...data.folders.map(f => new Option(f, f)));
      folder.value = selected;
      results.replaceChildren(...data.items.map(toolCard));
      if (!data.items.length) empty('No tools match these filters', 'Clear the search or choose a different folder.');
      document.getElementById('result-count').textContent = `${data.total} matching tools · ${data.total_catalog} in your library${favoritesOnly ? ' · Favorites' : ''}`;
      page = data.page; pages = data.pages;
      document.getElementById('page-number').textContent = `Page ${page} of ${pages}`;
      document.getElementById('pagination').hidden = pages <= 1;
      document.getElementById('previous').disabled = page <= 1;
      document.getElementById('next').disabled = page >= pages;
      if (data.errors.length) report(new Error(`Some manifests need attention: ${data.errors.join('; ')}`));
      results.setAttribute('aria-busy', 'false');
    } catch (error) { report(error); }
  }
  if (mode === 'catalog') {
    let timer;
    document.getElementById('search').addEventListener('input', () => { clearTimeout(timer); page = 1; timer = setTimeout(loadLibrary, 180); });
    ['folder', 'status'].forEach(id => document.getElementById(id).addEventListener('change', () => { page = 1; loadLibrary(); }));
    document.getElementById('reset').addEventListener('click', () => {
      ['search', 'folder', 'status'].forEach(id => document.getElementById(id).value = '');
      favoritesOnly = false; page = 1; loadLibrary();
    });
    document.getElementById('previous').addEventListener('click', () => { page--; loadLibrary(); });
    document.getElementById('next').addEventListener('click', () => { page++; loadLibrary(); });
    document.getElementById('view-toggle').addEventListener('click', event => {
      const enabled = results.classList.toggle('list-view');
      event.currentTarget.textContent = enabled ? 'Grid view' : 'List view';
      event.currentTarget.setAttribute('aria-pressed', String(enabled));
    });
    document.addEventListener('keydown', event => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault(); document.getElementById('search').focus();
      }
    });
    loadLibrary();
  }

  const jobNodes = new Map();
  async function loadJobs() {
    try {
      const data = await api('/api/jobs');
      clearMessage();
      if (!data.items.length) {
        results.replaceChildren(); jobNodes.clear();
        empty('No jobs in this session', 'Run a tool from the library. Completed output is kept in memory, not in Git.');
      } else {
        if (!jobNodes.size) results.replaceChildren();
        const valid = new Set(data.items.map(j => j.id));
        for (const [id, node] of jobNodes) if (!valid.has(id)) { node.remove(); jobNodes.delete(id); }
        for (const job of data.items) {
          let card = jobNodes.get(job.id);
          if (!card) {
            card = element('article', undefined, 'panel job-card');
            const heading = element('div', undefined, 'job-heading');
            heading.append(element('h2', job.tool));
            const controls = element('div');
            controls.append(element('span', '', 'badge'));
            controls.append(button('Stop', async () => { try { await api(`/api/jobs/${job.id}/stop`, {}); await loadJobs(); } catch (e) { report(e); } }, 'quiet'));
            heading.append(controls);
            const detail = element('details'); detail.open = true;
            detail.append(element('summary', 'Output'), element('pre', ''));
            card.append(heading, detail); results.prepend(card); jobNodes.set(job.id, card);
          }
          const badge = card.querySelector('.badge'); badge.className = `badge ${job.status}`; badge.textContent = job.status;
          card.querySelector('button').disabled = !['running', 'queued'].includes(job.status);
          const output = card.querySelector('pre');
          const follow = output.scrollTop + output.clientHeight >= output.scrollHeight - 30;
          if (output.textContent !== job.output) { output.textContent = job.output || 'Waiting for output…'; if (follow) output.scrollTop = output.scrollHeight; }
        }
      }
      results.setAttribute('aria-busy', 'false');
    } catch (error) { report(error); }
  }
  if (mode === 'jobs') { loadJobs(); setInterval(() => { if (!document.hidden) loadJobs(); }, 2000); }

  function askUnlock(id) {
    activeUnlock = id; document.getElementById('vault-passphrase').value = '';
    unlockDialog.showModal(); document.getElementById('vault-passphrase').focus();
  }
  document.getElementById('unlock-form').addEventListener('submit', async event => {
    event.preventDefault();
    const id = activeUnlock, input = document.getElementById('vault-passphrase');
    const value = input.value; input.value = '';
    const submit = event.currentTarget.querySelector('[type=submit]'); submit.disabled = true;
    try {
      await api(`/api/vaults/${id}/unlock`, {passphrase: value});
      if (!unlockDialog.open || activeUnlock !== id || document.hidden) {
        await api(`/api/vaults/${id}/lock`, {}); return;
      }
      unlockDialog.close(); await loadVaults(); await openBundle(id, 1);
    } catch (error) { document.getElementById('unlock-error').textContent = error.message; }
    finally { submit.disabled = false; }
  });
  function privateForm(id, tool) {
    const form = element('form', undefined, 'private-tool');
    form.append(element('h3', tool.name), element('p', tool.description));
    const fields = element('div', undefined, 'form-grid');
    const controls = new Map();
    for (const p of tool.parameters) {
      const label = element('label', p.label || p.name);
      let input;
      if (p.type === 'choice') {
        input = element('select');
        input.append(...p.choices.map(c => new Option(c, c)));
        input.value = p.default || p.choices[0];
      } else {
        input = element('input'); input.type = p.type === 'boolean' ? 'checkbox' : (p.type === 'integer' ? 'number' : 'text');
        if (p.type === 'boolean') { input.checked = Boolean(p.default); label.className = 'check-label'; }
        else input.value = p.default ?? '';
        if (p.min !== undefined) input.min = p.min;
        if (p.max !== undefined) input.max = p.max;
      }
      input.required = Boolean(p.required); input.autocomplete = 'off';
      controls.set(p.name, input); label.append(input); fields.append(label);
    }
    form.append(fields);
    const consentLabel = element('label', 'I trust this code and understand it runs with my OS permissions.', 'check-label');
    const consent = element('input'); consent.type = 'checkbox'; consent.required = true; consentLabel.prepend(consent); form.append(consentLabel);
    const run = element('button', tool.availability === 'ready' ? 'Run locked tool' : 'Setup required');
    run.type = 'submit'; run.disabled = tool.availability !== 'ready'; form.append(run);
    const status = element('p'); status.setAttribute('role', 'status'); form.append(status);
    form.addEventListener('submit', async event => {
      event.preventDefault(); run.disabled = true;
      const values = {};
      for (const [name, input] of controls) values[name] = input.type === 'checkbox' ? input.checked : input.value;
      try { await api(`/api/vaults/${id}/tools/${tool.index}/run`, {values, confirm: consent.checked}); status.textContent = 'Queued. Open Jobs & output to follow execution.'; }
      catch (error) { status.textContent = error.message; }
      finally { run.disabled = tool.availability !== 'ready'; }
    });
    return form;
  }
  async function openBundle(id, selectedPage) {
    const current = ++privateGeneration;
    try {
      const data = await api(`/api/vaults/${id}/contents?page=${selectedPage}`);
      if (current !== privateGeneration || document.hidden) return;
      activeBundle = id;
      document.getElementById('bundle-name').textContent = data.name;
      const tools = document.getElementById('bundle-tools');
      tools.replaceChildren(...data.tools.map(t => privateForm(id, t)));
      if (!data.tools.length) tools.append(element('p', 'No executable tools on this page. Files are available below.'));
      const files = document.getElementById('bundle-files');
      files.replaceChildren(...data.files.map(file => {
        const row = element('div', undefined, 'file-row');
        const link = element('a', 'Download plaintext'); link.href = `/api/vaults/${id}/files/${file.index}`;
        row.append(element('span', file.name), link); return row;
      }));
      const pager = document.getElementById('bundle-pages'); pager.replaceChildren();
      const max = Math.max(data.file_pages, data.tool_pages || 1);
      if (max > 1) {
        const previous = button('Previous', () => openBundle(id, selectedPage - 1)); previous.disabled = selectedPage === 1;
        const next = button('Next', () => openBundle(id, selectedPage + 1)); next.disabled = selectedPage >= max;
        pager.append(previous, element('span', `Page ${selectedPage} of ${max}`), next);
      }
      if (!bundleDialog.open) bundleDialog.showModal();
    } catch (error) { hidePrivate(); report(error); }
  }
  async function lock(id) {
    hidePrivate(); results.replaceChildren();
    try { await api(`/api/vaults/${id}/lock`, {}); await loadVaults(); } catch (error) { report(error); }
  }
  async function loadVaults() {
    const current = privateGeneration;
    try {
      const data = await api('/api/vaults');
      if (current !== privateGeneration || document.hidden) return;
      if (activeBundle && !data.items.some(v => v.id === activeBundle && !v.locked)) hidePrivate();
      results.replaceChildren(...data.items.map(vault => {
        const card = element('article', undefined, 'tool-card locked-card');
        const top = element('div', undefined, 'card-top');
        top.append(element('span', vault.locked ? '••' : 'O', 'tool-icon locked-mark'), element('span', vault.locked ? 'Locked' : 'Unlocked', 'badge'));
        card.append(top, element('h2', vault.locked ? 'Encrypted bundle' : vault.name),
          element('p', vault.locked ? 'Name, type, tools, and filenames are encrypted.' : `${vault.kind === 'folder' ? 'Locked folder' : 'Locked tool'} · ${vault.seconds_left}s remaining in this lease.`));
        const controls = element('div', undefined, 'card-actions');
        if (vault.locked) controls.append(button('Unlock', () => askUnlock(vault.id)));
        else controls.append(button('Open bundle', () => openBundle(vault.id, 1)), button('Lock now', () => lock(vault.id), 'quiet'));
        card.append(controls, element('small', `Bundle ${vault.id.slice(0, 8)}`)); return card;
      }));
      if (!data.items.length) empty('No encrypted bundles yet', 'Create a locked tool or folder locally. Only ciphertext should be committed.', button('How to add a bundle', () => instructions.showModal()));
      results.setAttribute('aria-busy', 'false');
    } catch (error) { report(error); }
  }
  document.getElementById('lock-all')?.addEventListener('click', async event => {
    event.currentTarget.disabled = true; hidePrivate(); results.replaceChildren();
    try { await api('/api/vaults/lock-all', {}); await loadVaults(); } catch (error) { report(error); }
    finally { document.getElementById('lock-all').disabled = false; }
  });
  if (mode === 'vaults') { loadVaults(); setInterval(() => { if (!document.hidden) loadVaults(); }, 3000); }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { hidePrivate(); if (unlockDialog.open) unlockDialog.close(); if (mode === 'jobs') { results.replaceChildren(); jobNodes.clear(); } }
    else if (mode === 'vaults') loadVaults();
    else if (mode === 'jobs') loadJobs();
  });
})();
