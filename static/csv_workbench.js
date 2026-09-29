'use strict';
(() => {
  const $ = id => document.getElementById(id), status = $('csv-status');
  if (!status) return;
  let worker, serial = 0, latestView = 0, page = 1, colPage = 1, current = null, loaded = false, edited = false, filename = 'table';
  const pending = new Map();
  function note(text, error = false) { status.textContent = text; status.className = error ? 'notice error' : 'notice'; }
  try { worker = new Worker('/static/csv_worker.js'); }
  catch (_) { note('Web Workers are unavailable. This browser cannot run the workbench.', true); return; }
  worker.onmessage = event => {
    const entry = pending.get(event.data.id); if (!entry) return; pending.delete(event.data.id);
    if (event.data.error) entry.reject(new Error(event.data.error)); else entry.resolve(event.data.result);
  };
  worker.onerror = () => {
    for (const item of pending.values()) item.reject(new Error('CSV worker stopped. Reload this page and import again.'));
    pending.clear(); loaded = false; $('csv-controls').disabled = true; $('csv-export-controls').disabled = true;
    note('CSV worker failed; export disabled. Reload this page.', true);
  };
  function viewOptions() { return {query: $('csv-filter').value, header: $('csv-header').checked, page, colPage}; }
  function call(action, data = {}) {
    const id = ++serial;
    return new Promise((resolve, reject) => { pending.set(id, {resolve, reject}); worker.postMessage({id, action, data, view: viewOptions()}); });
  }
  function el(tag, text) { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; return node; }
  function editButton(value, row, column) {
    const button = el('button', value.length > 120 ? value.slice(0, 120) + '…' : value || '(empty)');
    button.type = 'button'; button.className = 'cell-button'; button.setAttribute('aria-label', `Edit row ${row + 1}, column ${column + 1}`);
    button.addEventListener('click', () => {
      $('csv-edit-title').textContent = `Row ${row + 1} · column ${column + 1}`;
      $('csv-edit-value').value = value; $('csv-edit-dialog').dataset.row = row; $('csv-edit-dialog').dataset.column = column;
      $('csv-edit-dialog').showModal(); $('csv-edit-value').focus();
    }); return button;
  }
  function render(data) {
    current = data; page = data.page; colPage = data.colPage;
    $('csv-controls').disabled = false; $('csv-export-controls').disabled = false;
    $('csv-count').textContent = `${data.count} total rows · ${data.width} columns · ${data.matches} matching data rows${data.ragged ? ' · Unequal row widths: blank cells are not silently normalized.' : ''}`;
    $('csv-sort-col').max = Math.max(1, data.width);
    $('csv-undo').disabled = data.undo === 0;
    $('csv-page').textContent = `${page} / ${data.pages}`; $('csv-col-page').textContent = `${colPage} / ${data.colPages}`;
    $('csv-prev').disabled = page <= 1; $('csv-next').disabled = page >= data.pages;
    $('csv-col-prev').disabled = colPage <= 1; $('csv-col-next').disabled = colPage >= data.colPages;
    const table = $('csv-grid'); table.replaceChildren();
    const thead = el('thead'), tr = el('tr'); tr.append(el('th', 'Row'));
    data.columns.forEach((column, index) => {
      const th = el('th'); th.scope = 'col';
      if (data.headings) th.append(editButton(data.headings[index] || '', 0, column)); else th.textContent = `Column ${column + 1}`;
      tr.append(th);
    }); thead.append(tr); table.append(thead);
    const body = el('tbody');
    data.rows.forEach(row => {
      const tr = el('tr'); tr.append(el('th', String(row.index + 1)));
      data.columns.forEach((column, index) => { const cell = el('td'); cell.append(editButton(row.cells[index] || '', row.index, column)); tr.append(cell); }); body.append(tr);
    }); table.append(body);
  }
  async function refresh(action = 'view', data = {}) {
    const version = ++latestView;
    try {
      const result = await call(action, data);
      if (version !== latestView) return;
      render(result); note(`Ready · delimiter ${result.delimiter === '\t' ? 'tab' : result.delimiter}. Changes are in browser memory only.`); return true;
    } catch (error) { if (version === latestView) note(error.message, true); }
  }
  $('csv-import').addEventListener('click', async () => {
    const file = $('csv-file').files[0]; if (!file) { note('Select a CSV file first.', true); return; }
    if (edited && !window.confirm('Discard this table’s edits and reload the selected file?')) return;
    loaded = false; edited = false; page = colPage = 1; $('csv-filter').value = '';
    $('csv-controls').disabled = true; $('csv-export-controls').disabled = true; $('csv-grid').replaceChildren();
    $('csv-import').disabled = true; note('Reading and parsing locally…');
    const version = ++latestView;
    try {
      const result = await call('load', {file, encoding: $('import-encoding').value, delimiter: $('import-delimiter').value, quote: $('import-quote').value});
      if (version !== latestView) return;
      filename = file.name.replace(/\.[^.]*$/, '') || 'table'; loaded = true; render(result);
      note(`Imported locally · delimiter ${result.delimiter === '\t' ? 'tab' : result.delimiter}. Check auto-detection before editing.`);
    } catch (error) { note(error.message, true); }
    finally { $('csv-import').disabled = false; }
  });
  let filterTimer;
  $('csv-filter').addEventListener('input', () => { page = 1; clearTimeout(filterTimer); filterTimer = setTimeout(() => refresh(), 180); });
  $('csv-header').addEventListener('change', () => { page = 1; refresh(); });
  $('csv-prev').addEventListener('click', () => { page--; refresh(); });
  $('csv-next').addEventListener('click', () => { page++; refresh(); });
  $('csv-col-prev').addEventListener('click', () => { colPage--; refresh(); });
  $('csv-col-next').addEventListener('click', () => { colPage++; refresh(); });
  $('csv-edit-cancel').addEventListener('click', () => $('csv-edit-dialog').close());
  $('csv-edit-form').addEventListener('submit', async event => {
    event.preventDefault(); edited = true;
    const saved = await refresh('edit', {row: Number($('csv-edit-dialog').dataset.row), column: Number($('csv-edit-dialog').dataset.column), value: $('csv-edit-value').value});
    if (saved) $('csv-edit-dialog').close();
  });
  $('csv-undo').addEventListener('click', () => { edited = true; refresh('undo'); });
  $('csv-reset').addEventListener('click', () => { if (window.confirm('Restore the originally imported table? This action can be undone within the history budget.')) { edited = true; refresh('reset'); } });
  $('csv-transform-apply').addEventListener('click', () => {
    edited = true; refresh('transform', {kind: $('csv-transform').value, column: Number($('csv-sort-col').value) - 1});
  });
  async function download(format) {
    if (!loaded) return;
    try {
      const result = await call('export', {format, filtered: $('csv-filtered-export').checked, safe: $('safe-formulas').checked,
        delimiter: $('export-delimiter').value, quote: $('export-quote').value});
      const url = URL.createObjectURL(new Blob([result.text], {type: format === 'json' ? 'application/json' : 'text/csv;charset=utf-8'}));
      const link = el('a'); link.href = url; link.download = `${filename}-edited.${format}`; document.body.append(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000); note(`Exported ${result.rows} rows as ${format.toUpperCase()}. The original file is unchanged.`);
    } catch (error) { note(error.message, true); }
  }
  $('csv-export').addEventListener('click', () => download('csv'));
  $('csv-json').addEventListener('click', () => download('json'));
  window.addEventListener('beforeunload', event => { if (edited) { event.preventDefault(); event.returnValue = ''; } });
})();
