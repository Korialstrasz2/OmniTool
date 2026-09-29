'use strict';
(() => {
  const root = document.getElementById('file-workbench');
  if (!root) return;
  const api = window.OmniAPI, $ = id => document.getElementById(id);
  const message = $('file-message');
  const tell = (text, error = false) => { message.textContent = text; message.className = `notice${error ? ' error' : ''}`; message.hidden = false; };
  const fail = e => tell(e.message || String(e), true);
  const node = (tag, text) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; };
  const row = values => { const tr = node('tr'); for (const value of values) tr.append(node('td', value)); return tr; };
  const button = (text, fn) => { const b = node('button', text); b.type = 'button'; b.className = 'secondary'; b.addEventListener('click', fn); return b; };
  const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
  const release = token => { if (token) api(`/api/files/leases/${token}/release`, {}).catch(() => {}); };
  let task = null, polling = 0;
  async function disposeTask() {
    if (task) { const old = task; task = null; polling++; await api(`/api/files/tasks/${old}/dismiss`, {}).catch(() => {}); }
  }
  async function follow(id) {
    task = id; const current = ++polling;
    if ($('task-cancel')) $('task-cancel').hidden = false;
    for (;;) {
      const data = await api(`/api/files/tasks/${id}`);
      if (current !== polling) throw new Error('Operation view changed. Its result was not applied to this page.');
      tell(`${data.action}: ${data.state}`);
      if (!['queued', 'running'].includes(data.state)) {
        if ($('task-cancel')) $('task-cancel').hidden = true;
        if (data.state !== 'succeeded') throw new Error(data.error || 'Operation canceled. Check for partial output.');
        return data.result;
      }
      await pause(750);
    }
  }
  $('task-cancel')?.addEventListener('click', async () => {
    try { if (task) await api(`/api/files/tasks/${task}/dismiss`, {}); tell('Stop requested. A published output folder is kept; a staging folder may remain.'); }
    catch (e) { fail(e); }
  });

  if (root.dataset.mode === 'dual') {
    let inventoryToken = null, previewToken = null, pairs = [], selected = {left: null, right: null};
    let pages = {left: 1, right: 1}, versions = {left: 0, right: 0}, epoch = 0;
    function invalidatePreview() { release(previewToken); previewToken = null; $('dual-review').hidden = true; $('dual-confirm').value = ''; }
    function invalidateFolders() {
      epoch++; release(inventoryToken); inventoryToken = null; invalidatePreview();
      pairs = []; selected = {left: null, right: null}; $('dual-matching').hidden = true;
      for (const side of ['left', 'right']) {
        versions[side]++; $(side + '-thumb').hidden = true; $(side + '-thumb').removeAttribute('src');
        $(side + '-selected').textContent = 'No file selected'; $(side + '-thumb-note').textContent = '';
      }
    }
    function staged() {
      invalidatePreview(); $('staged-pairs').replaceChildren();
      pairs.forEach((pair, index) => {
        const tr = row([pair.left.path, pair.right.path]);
        const td = node('td'); td.append(button('Remove', () => { pairs.splice(index, 1); staged(); }));
        tr.append(td); $('staged-pairs').append(tr);
      });
      $('preview-pairs').disabled = !pairs.length;
    }
    async function listing(side) {
      if (!inventoryToken) return;
      const version = ++versions[side], token = inventoryToken;
      try {
        const data = await api(`/api/files/dual/${token}/list/${side}?page=${pages[side]}&query=${encodeURIComponent($(side + '-query').value)}`);
        if (version !== versions[side] || token !== inventoryToken) return;
        pages[side] = data.page; $(side + '-page').textContent = `${data.total} files · ${data.page}/${data.pages}`;
        $(side + '-prev').disabled = data.page <= 1; $(side + '-next').disabled = data.page >= data.pages;
        $(side + '-files').replaceChildren(...data.items.map(item => {
          const b = button('', () => {
            selected[side] = item; $(side + '-selected').textContent = item.path;
            const img = $(side + '-thumb'); img.hidden = false;
            $(side + '-thumb-note').textContent = 'Loading selected-file preview…';
            img.onload = () => { $(side + '-thumb-note').textContent = ''; };
            img.onerror = () => { img.hidden = true; $(side + '-thumb-note').textContent = 'Preview unavailable. You can still match this file by name.'; };
            img.src = `/api/files/dual/${token}/thumbnail/${side}/${item.index}`;
            $('stage-pair').disabled = !(selected.left && selected.right);
            listing(side);
          });
          b.className = 'file-choice'; b.setAttribute('aria-pressed', String(selected[side]?.index === item.index));
          b.append(node('span', item.path), node('small', `${item.bytes.toLocaleString()} B`)); return b;
        }));
        if (!data.items.length) $(side + '-files').append(node('p', 'No matching files.'));
      } catch (e) { fail(e); }
    }
    for (const side of ['left', 'right']) {
      $(side + '-root').addEventListener('input', invalidateFolders);
      let timer;
      $(side + '-query').addEventListener('input', () => { clearTimeout(timer); pages[side] = 1; timer = setTimeout(() => listing(side), 180); });
      $(side + '-prev').addEventListener('click', () => { pages[side]--; listing(side); });
      $(side + '-next').addEventListener('click', () => { pages[side]++; listing(side); });
    }
    $('dual-open').addEventListener('submit', async e => {
      e.preventDefault(); invalidateFolders(); const current = epoch;
      const b = e.currentTarget.querySelector('button'); b.disabled = true;
      try {
        const data = await api('/api/files/dual/open', {left: $('left-root').value, right: $('right-root').value});
        if (current !== epoch) { release(data.token); return; }
        inventoryToken = data.token; pages = {left: 1, right: 1}; staged(); $('stage-pair').disabled = true;
        $('dual-matching').hidden = false;
        await Promise.all([listing('left'), listing('right')]);
        tell(`Folders loaded. Skipped: ${data.skipped.left} left, ${data.skipped.right} right. No files changed.`);
      } catch (error) { fail(error); } finally { b.disabled = false; }
    });
    $('stage-pair').addEventListener('click', () => {
      if (!selected.left || !selected.right) return;
      if (pairs.some(p => p.left.index === selected.left.index || p.right.index === selected.right.index)) {
        tell('Each reference and target can be used once. Remove the existing pair first.', true); return;
      }
      if (pairs.length >= 500) { tell('Maximum 500 staged mappings.', true); return; }
      pairs.push({left: {...selected.left}, right: {...selected.right}}); staged();
    });
    $('preview-pairs').addEventListener('click', async () => {
      const token = inventoryToken, signature = JSON.stringify(pairs);
      $('preview-pairs').disabled = true;
      try {
        const data = await api('/api/files/dual/preview', {token, pairs: pairs.map(p => ({left: p.left.index, right: p.right.index}))});
        if (token !== inventoryToken || signature !== JSON.stringify(pairs)) { release(data.token); return; }
        invalidatePreview(); previewToken = data.token;
        $('dual-plan').replaceChildren(...data.rows.map(r => row([r.source, r.target])));
        $('dual-summary').textContent = `${data.count} files will be renamed. Unchanged names are omitted.`;
        $('dual-confirm').placeholder = `RENAME ${data.count}`;
        $('dual-conflicts').hidden = !data.conflicts.length;
        $('dual-conflicts').textContent = data.conflicts.map(c => `${c.path} → ${c.target}: ${c.reason}`).join('\n');
        $('dual-run').disabled = !!data.conflicts.length || !data.count;
        $('dual-review').hidden = false;
      } catch (e) { fail(e); } finally { $('preview-pairs').disabled = !pairs.length; }
    });
    $('dual-apply').addEventListener('submit', async e => {
      e.preventDefault(); $('dual-run').disabled = true;
      try {
        const data = await api('/api/files/dual/apply', {token: previewToken, confirmation: $('dual-confirm').value});
        invalidateFolders(); tell(`Renamed ${data.count} files. Operation ${data.id}. Undo is available below.`);
      } catch (error) { invalidatePreview(); fail(error); }
      finally { await operations(); }
    });
    async function operations() {
      try { const data = await api('/api/maintenance/rename/operations');
        $('rename-operation').replaceChildren(...data.items.map(op => new Option(`${op.status} · ${op.root} · ${op.id}`, op.id)));
      } catch (e) { fail(e); }
    }
    $('rename-recover').addEventListener('submit', async e => {
      e.preventDefault(); const b = e.currentTarget.querySelector('button'); b.disabled = true;
      try { const data = await api('/api/maintenance/rename/recover', {operation: $('rename-operation').value, confirmation: $('undo-confirm').value});
        invalidateFolders(); $('undo-confirm').value = ''; tell(`Operation ${data.id}: ${data.status}.`); await operations();
      } catch (error) { fail(error); } finally { b.disabled = false; }
    });
    operations();
  }

  if (root.dataset.mode === 'compare') {
    let reportTask = null, page = 1, reportVersion = 0;
    async function report() {
      if (!reportTask) return;
      const id = reportTask, version = ++reportVersion;
      try {
        const data = await api(`/api/files/tasks/${id}?page=${page}&query=${encodeURIComponent($('report-query').value)}&status=${encodeURIComponent($('report-status').value)}`);
        if (id !== reportTask || version !== reportVersion) return;
        const value = data.result;
        if (!value) return;
        $('compare-rows').replaceChildren(...value.rows.map(r => row([r.path, r.status, r.left.join(' | '), r.right.join(' | ')])));
        if (!value.rows.length) $('compare-rows').append(row(['No matching rows', '', '', '']));
        page = value.page; $('report-page').textContent = `${value.total} rows · Page ${page}/${value.pages}`;
        $('report-prev').disabled = page <= 1; $('report-next').disabled = page >= value.pages;
      } catch (e) { fail(e); }
    }
    $('compare-form').addEventListener('submit', async e => {
      e.preventDefault(); $('compare-start').disabled = true; $('comparison-report').hidden = true; reportTask = null;
      try {
        await disposeTask();
        const data = await api('/api/files/compare', {left: $('compare-left').value, right: $('compare-right').value,
          mode: $('compare-mode').value, recursive: $('recursive').checked, hash_budget_mib: Number($('hash-budget').value)});
        const result = await follow(data.task); reportTask = data.task;
        $('compare-summary').textContent = `${result.left_root} ↔ ${result.right_root}\nMode: ${result.mode}. ${Object.entries(result.counts).map(([k,v]) => `${k}: ${v}`).join(' · ')}`;
        const skipped = result.skipped.left.length + result.skipped.right.length;
        $('compare-scope').textContent = `${skipped} skipped paths. ${result.mode === 'content' ? 'Content matches use size and SHA-256.' : 'This mode does NOT establish equal contents.'} Export JSON for complete scope details.`;
        $('report-status').replaceChildren(new Option('All statuses', ''), ...Object.keys(result.counts).map(k => new Option(k, k)));
        $('report-query').value = ''; page = 1;
        $('report-json').href = `/api/files/tasks/${data.task}/export/json`; $('report-csv').href = `/api/files/tasks/${data.task}/export/csv`;
        $('comparison-report').hidden = false; await report(); tell('Comparison complete. Neither folder was modified. Reports expire after ten minutes.');
      } catch (error) { fail(error); } finally { $('compare-start').disabled = false; $('task-cancel').hidden = true; }
    });
    let timer;
    $('report-query').addEventListener('input', () => { clearTimeout(timer); page = 1; timer = setTimeout(report, 180); });
    $('report-status').addEventListener('change', () => { page = 1; report(); });
    $('report-prev').addEventListener('click', () => { page--; report(); });
    $('report-next').addEventListener('click', () => { page++; report(); });
  }

  if (root.dataset.mode === 'convert') {
    let inspectTask = null, version = 0;
    const inputs = ['convert-input', 'convert-out', 'convert-dpi', 'convert-pages'];
    for (const id of inputs) $(id).addEventListener('input', () => { version++; inspectTask = null; $('conversion-review').hidden = true; $('convert-confirm').value = ''; });
    $('convert-form').addEventListener('submit', async e => {
      e.preventDefault(); const current = version; $('inspect-start').disabled = true; inspectTask = null;
      $('conversion-review').hidden = true; $('conversion-result').hidden = true;
      try {
        await disposeTask();
        const data = await api('/api/files/convert/preview', {input: $('convert-input').value, out: $('convert-out').value,
          dpi: Number($('convert-dpi').value), max_pages: Number($('convert-pages').value)});
        const result = await follow(data.task);
        if (current !== version) { tell('Inputs changed. Inspect the updated configuration.'); return; }
        inspectTask = data.task;
        $('conversion-summary').textContent = `${result.kind.toUpperCase()} · ${result.selected_pages}/${result.total_pages} pages · ${result.decoded_bytes.toLocaleString()} decoded bytes. Input SHA-256: ${result.sha256}`;
        $('conversion-omitted').hidden = !result.omitted_pages;
        $('conversion-omitted').textContent = `${result.omitted_pages} trailing pages will NOT be converted because of the selected page limit.`;
        $('conversion-images').replaceChildren(...result.images.map(i => row([i.filename, `${i.width} × ${i.height}`])));
        $('convert-confirm').value = ''; $('convert-confirm').placeholder = `CONVERT ${result.selected_pages}`;
        $('convert-run').disabled = false; $('conversion-review').hidden = false;
        tell('Inspection complete. No output folder has been created.');
      } catch (error) { fail(error); } finally { $('inspect-start').disabled = false; $('task-cancel').hidden = true; }
    });
    $('convert-apply').addEventListener('submit', async e => {
      e.preventDefault(); if (!inspectTask) return;
      $('convert-run').disabled = true; $('inspect-start').disabled = true;
      const preview = inspectTask;
      try {
        const data = await api('/api/files/convert/apply', {task: preview, confirmation: $('convert-confirm').value});
        inspectTask = null; const result = await follow(data.task);
        $('conversion-output').textContent = `Created ${result.selected_pages} PNG files in ${result.output}`;
        $('conversion-json').href = `/api/files/tasks/${data.task}/export/json`;
        $('conversion-result').hidden = false; $('conversion-review').hidden = true;
        await api(`/api/files/tasks/${preview}/dismiss`, {}).catch(() => {});
        tell('Conversion completed. The source file and existing output folders were not modified.');
      } catch (error) { fail(error); $('convert-run').disabled = !inspectTask; }
      finally { $('inspect-start').disabled = false; $('task-cancel').hidden = true; }
    });
  }
})();
