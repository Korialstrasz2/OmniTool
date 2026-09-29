'use strict';
(() => {
  const app = document.getElementById('content-app'); if (!app) return;
  const api = window.OmniAPI, status = document.getElementById('content-status'), stop = document.getElementById('stop-content');
  let active = null, busy = false, epoch = 0, scanTask = null, previewTask = null;
  const el = (tag, text) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; return node; };
  const say = (text, error = false) => { status.textContent = text; status.className = error ? 'notice error' : 'notice'; };
  const byId = id => document.getElementById(id);
  function setBusy(value) {
    busy = value; document.querySelectorAll('.content-config').forEach(f => f.disabled = value);
    byId('check-backend')?.toggleAttribute('disabled', value); stop.disabled = !value;
  }
  async function run(route, payload, done) {
    if (busy) return;
    const generation = ++epoch; setBusy(true); say('Queued. This operation uses bounded local worker resources.');
    try {
      const started = await api(route, payload);
      if (generation !== epoch) { await api(`/api/content/tasks/${started.task}/stop`, {}); return; }
      active = started.task;
      while (generation === epoch) {
        const item = await api(`/api/content/tasks/${active}`);
        if (generation !== epoch) return;
        if (item.state === 'succeeded') { say('Completed.'); await done(item.result, active); break; }
        if (['failed', 'canceled'].includes(item.state)) throw new Error(item.error || 'Operation canceled');
        say(item.state === 'queued' ? 'Queued behind another local task…' : 'Working. Originals are not modified.');
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    } catch (error) { if (generation === epoch) say(error.message, true); }
    finally { if (generation === epoch) { active = null; setBusy(false); } }
  }
  stop.addEventListener('click', async () => {
    if (!active) return;
    stop.disabled = true;
    try { await api(`/api/content/tasks/${active}/stop`, {}); say('Stop requested. A local model may continue its own generation; check partial output folders after file operations.'); }
    catch (error) { say(error.message, true); }
  });
  function invalidate() { previewTask = null; if (byId('lyrics-review')) byId('lyrics-review').hidden = true; }
  byId('clear-content').addEventListener('click', async () => {
    ++epoch; active = null;
    try { await api('/api/content/clear', {}); say('Session results cleared. Running workers were asked to stop; exported files remain on disk.'); }
    catch (error) { say(error.message, true); }
    setBusy(false); invalidate(); scanTask = null;
    if (byId('lyrics-selection')) { byId('lyrics-selection').hidden = true; byId('lyrics-tracks').replaceChildren(); byId('lyrics-preview-rows').replaceChildren(); }
    if (byId('prompt-result')) byId('prompt-result').value = '';
  });
  function download(text, filename) {
    const url = URL.createObjectURL(new Blob([text], {type: 'text/plain;charset=utf-8'}));
    const link = el('a'); link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  if (app.dataset.tool === 'prompts') {
    const originalInstruction = byId('prompt-system').value;
    byId('prompt-preset').addEventListener('change', event => {
      if (event.target.value === 'image') byId('prompt-system').value = originalInstruction;
      else if (event.target.value === 'rewrite') byId('prompt-system').value = 'Rewrite the user text clearly while preserving its meaning. Do not invent facts. Return only the rewritten text.';
    });
    byId('prompt-form').addEventListener('submit', event => {
      event.preventDefault();
      run('/api/content/prompts/generate', {idea: byId('prompt-idea').value, system: byId('prompt-system').value,
        max_tokens: Number(byId('prompt-tokens').value), temperature: Number(byId('prompt-temperature').value)},
      result => { byId('prompt-result').value = result.content; say(`Generated using ${result.kind} at ${result.url}. Result is editable and has not been saved.`); });
    });
    byId('check-backend').addEventListener('click', () => run('/api/content/prompts/status', {}, result => say(`Connected: ${result.model} · ${result.url}`)));
    byId('copy-prompt').addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(byId('prompt-result').value); say('Copied as plain text.'); }
      catch (_) { byId('prompt-result').select(); say('Clipboard access unavailable. The text is selected for manual copying.'); }
    });
    byId('export-prompt').addEventListener('click', () => download(byId('prompt-result').value, 'omnitool-prompt.txt'));
    return;
  }
  byId('lyrics-root').addEventListener('input', () => { scanTask = null; invalidate(); byId('lyrics-selection').hidden = true; });
  byId('lyrics-scan-form').addEventListener('submit', event => {
    event.preventDefault(); invalidate(); scanTask = null; byId('lyrics-selection').hidden = true;
    run('/api/content/lyrics/scan', {root: byId('lyrics-root').value}, (result, token) => {
      scanTask = token; const body = byId('lyrics-tracks'); body.replaceChildren();
      for (const track of result.tracks) {
        const row = el('tr'); row.dataset.index = track.index;
        const pick = el('td'), checkbox = el('input'); checkbox.type = 'checkbox'; checkbox.className = 'track-choice';
        checkbox.setAttribute('aria-label', `Select ${track.file}`); pick.append(checkbox);
        const file = el('td', track.file); file.append(el('small', ` · ${track.format} · ${track.duration}s${track.inferred ? ' · metadata inferred' : ''}`));
        const metadata = el('td');
        for (const key of ['artist', 'title', 'album']) {
          const label = el('label', key[0].toUpperCase() + key.slice(1)), input = el('input');
          input.type = 'text'; input.value = track[key]; input.dataset.key = key; input.maxLength = 1024; input.autocomplete = 'off';
          label.append(input); metadata.append(label);
        }
        row.append(pick, file, metadata, el('td', track.has_lyrics ? 'Protected by default' : 'None')); body.append(row);
      }
      byId('lyrics-warnings').textContent = result.warnings.map(w => `${w.path}: ${w.reason}`).join('\n') || 'No exclusions reported.';
      byId('lyrics-selection').hidden = false;
      say(`Inspected ${result.tracks.length} tracks. ${result.warning_count} warnings/exclusions. Select tracks explicitly; no audio changed.`);
    });
  });
  byId('lyrics-selection').addEventListener('input', invalidate);
  byId('lyrics-selection').addEventListener('change', invalidate);
  byId('lyrics-prepare').addEventListener('click', () => {
    if (!scanTask) { say('Scan the current folder first.', true); return; }
    invalidate();
    const selections = [...byId('lyrics-tracks').querySelectorAll('tr')].filter(row => row.querySelector('.track-choice').checked).map(row => {
      const item = {index: Number(row.dataset.index)};
      row.querySelectorAll('[data-key]').forEach(input => item[input.dataset.key] = input.value); return item;
    });
    if (!selections.length) { say('Select at least one track.', true); return; }
    run('/api/content/lyrics/prepare', {scan_task: scanTask, selections, out: byId('lyrics-out').value,
      provider: byId('lyrics-provider').value, replace: byId('lyrics-replace').checked,
      external_consent: byId('lyrics-external').checked}, (result, token) => {
      previewTask = token; byId('lyrics-confirm').value = '';
      byId('lyrics-review-summary').textContent = `${result.count} tagged copies will be written to ${result.output}. Review every ready track, then type WRITE ${result.count}. Skipped tracks are not copied.`;
      const rows = byId('lyrics-preview-rows'); rows.replaceChildren();
      result.rows.forEach(row => {
        const detail = el('details'); detail.className = 'lyric-review';
        detail.append(el('summary', `${row.file} · ${row.status}${row.reason ? ' · ' + row.reason : ''}`));
        if (row.status === 'ready') { detail.append(el('p', `Source: ${row.provider}. Plain lyrics embedded; ${row.synced ? 'an LRC sidecar is also saved.' : 'no synchronized sidecar.'}`), el('pre', row.plain)); }
        rows.append(detail);
      });
      byId('lyrics-review').hidden = false; say('Lyrics prepared for review. No audio or output directory has been written.');
    });
  });
  byId('lyrics-apply').addEventListener('click', () => {
    if (!previewTask) { say('Prepare a new review first.', true); return; }
    run('/api/content/lyrics/apply', {preview_task: previewTask, confirmation: byId('lyrics-confirm').value}, result => {
      invalidate(); say(`Created ${result.count} tagged copies in ${result.output}. Originals are unchanged.`);
    });
  });
})();
