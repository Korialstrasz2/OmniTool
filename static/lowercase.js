'use strict';
(() => {
  const $ = id => document.getElementById(id), api = window.OmniAPI;
  let preview = null, busy = false;
  const say = text => { $('rename-message').textContent = text; };
  function render(data) {
    preview = data; $('rename-rows').replaceChildren();
    for (const item of data.rows) {
      const row = document.createElement('tr');
      for (const key of ['source', 'target']) { const cell = document.createElement('td'); cell.textContent = item[key]; row.append(cell); }
      $('rename-rows').append(row);
    }
    $('rename-summary').textContent = `${data.count} proposed renames · ${data.scanned} entries inspected · ${data.skipped_count} skipped`;
    $('rename-page').textContent = `Page ${data.page} of ${data.pages}`;
    $('rename-previous').disabled = data.page === 1; $('rename-next').disabled = data.page === data.pages;
    $('rename-warnings').hidden = !(data.conflicts.length || data.skipped.length);
    $('rename-warnings').textContent = [...data.conflicts.map(c => `BLOCKED: ${c.path}: ${c.reason}`),
      ...data.skipped.map(s => `SKIPPED: ${s.path}: ${s.reason}`)].join('\n');
    $('rename-apply').disabled = !data.count || data.conflicts.length > 0;
    $('rename-instruction').textContent = data.conflicts.length ? 'Resolve collisions, then preview again. No files will be changed.' : `Type RENAME ${data.count} to apply this exact preview.`;
  }
  async function operations() {
    const data = await api('/api/maintenance/rename/operations');
    $('rename-operation').replaceChildren(new Option('Select an operation', ''), ...data.items.map(x => new Option(`${x.status} · ${x.root} · ${x.id}`, x.id)));
  }
  async function work(fn) {
    if (busy) return; busy = true;
    const controls = [...document.querySelectorAll('#main input,#main button,#main select')];
    const states = controls.map(x => x.disabled); controls.forEach(x => { x.disabled = true; });
    try { await fn(); } catch (error) { say(error.message); }
    finally {
      controls.forEach((x,i) => { x.disabled = states[i]; }); busy = false;
      if (preview) render(preview); else $('rename-apply').disabled = true;
    }
  }
  $('rename-root').addEventListener('input', () => { preview = null; $('rename-apply').disabled = true; $('rename-confirm').value = ''; });
  $('rename-preview').addEventListener('click', () => work(async () => {
    preview = null; $('rename-confirm').value = ''; say('Inspecting filenames; no files are being changed.');
    render(await api('/api/maintenance/rename/preview', {root: $('rename-root').value})); say('Preview complete. Review every page and any skipped entries.');
  }));
  for (const [id, delta] of [['rename-previous',-1],['rename-next',1]]) $(id).addEventListener('click', () => work(async () => {
    if (preview) render(await api(`/api/maintenance/rename/preview/${preview.token}?page=${preview.page+delta}`));
  }));
  $('rename-apply').addEventListener('click', () => work(async () => {
    if (!preview) throw new Error('Create a preview first.');
    if ($('rename-confirm').value !== `RENAME ${preview.count}`) throw new Error(`Type RENAME ${preview.count} exactly.`);
    const token = preview.token; preview = null; say('Applying the reviewed plan. Leave this window open; recovery journals are being recorded.');
    try {
      const result = await api('/api/maintenance/rename/apply', {token, confirmation: $('rename-confirm').value});
      say(`${result.count} filenames renamed. Operation ${result.id}.`); $('rename-result').textContent = `Applied: ${result.id}`;
    } finally { await operations(); $('rename-confirm').value = ''; }
  }));
  $('rename-recover').addEventListener('click', () => work(async () => {
    const result = await api('/api/maintenance/rename/recover', {operation: $('rename-operation').value, confirmation: $('rename-undo-confirm').value});
    preview = null; $('rename-undo-confirm').value = ''; say(`Original names restored for operation ${result.id}.`);
    $('rename-result').textContent = `Restored: ${result.id}`; await operations();
  }));
  $('rename-refresh').addEventListener('click', () => work(operations));
  operations().catch(error => say(error.message));
})();
