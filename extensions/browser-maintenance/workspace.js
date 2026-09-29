'use strict';
(async () => {
  const api = globalThis.browser || globalThis.chrome, core = OmniBrowserCore;
  const $ = id => document.getElementById(id);
  const firefox = typeof api.runtime.getBrowserInfo === 'function', engine = firefox ? 'firefox' : 'chromium';
  let profile, kind = 'history', rows = [], selected = new Set(), page = 1;
  let prepared = null, verified = null, restoration = null, busy = false, cancelRequested = false;
  const say = text => { $('message').textContent = text; };
  function counts() { $('selection-summary').textContent = `${rows.length} records previewed · ${selected.size} selected`; }
  function invalidateBackup() {
    prepared = null; verified = null; $('delete').disabled = true; $('verify').disabled = true;
    $('backup-status').textContent = 'Selection changed: create and verify a new backup.';
    $('delete-confirm').value = ''; $('verify-file').value = '';
    $('backup').disabled = selected.size === 0; counts();
  }
  function reset() {
    rows = []; selected.clear(); page = 1; invalidateBackup(); render();
  }
  function cell(tag, text) { const node = document.createElement(tag); node.textContent = text; return node; }
  function render() {
    const headings = kind === 'history' ? ['Select', 'URL', 'Title'] : ['Select', 'Host', 'Name', 'Path', 'Store / partition'];
    const tr = document.createElement('tr'); headings.forEach(h => tr.append(cell('th', h))); $('head').replaceChildren(tr);
    $('rows').replaceChildren();
    rows.slice((page - 1) * 50, page * 50).forEach((row, offset) => {
      const index = (page - 1) * 50 + offset, line = document.createElement('tr'), td = document.createElement('td');
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = selected.has(index);
      checkbox.setAttribute('aria-label', `Select record ${index + 1}`);
      checkbox.addEventListener('change', () => { checkbox.checked ? selected.add(index) : selected.delete(index); invalidateBackup(); });
      td.append(checkbox); line.append(td);
      const values = kind === 'history' ? [row.url, row.title || ''] : [row.domain, row.name, row.path,
        `${row.storeId} / ${row.partitionKey ? JSON.stringify(row.partitionKey) : 'unpartitioned'}${row.firstPartyDomain ? ' / ' + row.firstPartyDomain : ''}`];
      values.forEach(v => line.append(cell('td', v))); $('rows').append(line);
    });
    const pages = Math.max(1, Math.ceil(rows.length / 50));
    $('page-label').textContent = `Page ${page} of ${pages}`;
    $('previous').disabled = page === 1; $('next').disabled = page >= pages; counts();
  }
  async function work(fn) {
    if (busy) return;
    busy = true; cancelRequested = false; $('controls').disabled = true; $('clear').disabled = true;
    try { await fn(); } catch (error) { say(error.message || 'Action failed.'); }
    finally { busy = false; $('controls').disabled = false; $('clear').disabled = false; $('stop').disabled = true; }
  }
  function bind(id, fn) { $(id).addEventListener('click', () => work(fn)); }
  async function fileText(id) {
    const file = $(id).files[0];
    if (!file || file.size > 12 * 1024 * 1024) throw new Error('Choose a browser backup smaller than 12 MiB.');
    return file.text();
  }
  const allow = () => $('allowlist').value.split(/\r?\n/).map(s => s.trim()).filter(Boolean).map(core.host);
  function showResults(report) {
    $('results').textContent = report.map(r => `Record ${r.index + 1}: ${r.status}`).join('\n') || 'No records processed.';
  }
  try {
    const local = await api.storage.local.get(['profileId', 'protectedHosts', 'protectSubdomains']);
    profile = local.profileId || crypto.randomUUID();
    if (!local.profileId) await api.storage.local.set({profileId: profile});
    $('allowlist').value = (local.protectedHosts || []).join('\n');
    $('allow-subdomains').checked = Boolean(local.protectSubdomains);
  } catch (_) { $('controls').disabled = true; say('Cannot initialize local profile identity; maintenance is disabled.'); return; }
  for (const value of ['history', 'cookies']) $(value + '-tab').addEventListener('click', () => {
    if (busy) return;
    kind = value; reset();
    $('history-panel').hidden = kind !== 'history'; $('cookies-panel').hidden = kind !== 'cookies';
    $('history-tab').setAttribute('aria-pressed', String(kind === 'history'));
    $('cookies-tab').setAttribute('aria-pressed', String(kind === 'cookies'));
  });
  // Permission requests must be invoked directly from a user gesture.
  bind('history-access', async () => {
    const granted = await api.permissions.request({permissions: ['history']});
    say(granted ? 'History access granted for this profile. Enter a hostname and preview.' : 'History access was not granted.');
  });
  bind('cookies-access', async () => {
    const granted = await api.permissions.request({permissions: ['cookies'], origins: ['http://*/*', 'https://*/*']});
    if (!granted) throw new Error('Cookie access was not granted.');
    const stores = await api.cookies.getAllCookieStores();
    $('store').replaceChildren(new Option('Choose a cookie store', ''), ...stores.map(s => new Option(`Store ${s.id} (this profile)`, s.id)));
    say('Cookie access granted. Choose a store and review protected hostnames.');
  });
  bind('preview-history', async () => {
    reset(); const preview = await core.historyPreview(api, $('domain').value, $('history-subdomains').checked);
    rows = preview.rows; render();
    say(preview.truncated ? 'Search reached the 20,000-URL limit; this preview may be incomplete. Only explicitly selected URLs can be deleted.' : 'Review individual URLs. No records are selected automatically.');
  });
  bind('preview-cookies', async () => {
    reset(); const preview = await core.cookiePreview(api, $('store').value, allow(), $('allow-subdomains').checked, firefox);
    rows = preview.rows; render(); say(`${preview.protectedCount} cookies protected by your explicit allowlist. No cookie values are shown.`);
  });
  bind('save-allowlist', async () => {
    await api.storage.local.set({protectedHosts: allow(), protectSubdomains: $('allow-subdomains').checked});
    reset(); say('Protected hosts saved locally in this browser profile; create a new preview.');
  });
  for (const id of ['domain', 'history-subdomains', 'store', 'allowlist', 'allow-subdomains']) $(id).addEventListener('input', () => { if (!busy) reset(); });
  $('select-all').addEventListener('click', () => { rows.forEach((_, i) => selected.add(i)); invalidateBackup(); render(); });
  $('select-none').addEventListener('click', () => { selected.clear(); invalidateBackup(); render(); });
  $('previous').addEventListener('click', () => { page--; render(); });
  $('next').addEventListener('click', () => { page++; render(); });
  $('stop').addEventListener('click', () => { cancelRequested = true; say('Stopping after the current native API call. Completed changes remain applied.'); });
  $('clear').addEventListener('click', () => {
    if (busy) return;
    reset(); restoration = null; $('restore').disabled = true; $('restore-summary').textContent = '';
    document.querySelectorAll('input[type=password],input[type=file]').forEach(input => { input.value = ''; });
    $('results').textContent = 'In-memory preview cleared. Previously saved backups are unchanged.';
  });
  bind('backup', async () => {
    let password = $('new-password').value, repeated = $('repeat-password').value;
    $('new-password').value = ''; $('repeat-password').value = '';
    if (password !== repeated) throw new Error('Passphrases do not match.');
    prepared = null; verified = null; $('delete').disabled = true; $('verify').disabled = true;
    const data = await core.snapshot(api, kind, [...selected].sort((a, b) => a-b).map(i => rows[i]), profile, engine);
    const text = await core.seal(data, password), checked = await core.unseal(text, password);
    password = ''; repeated = ''; // Drops references, not a secure-memory-wipe guarantee.
    if (await core.digest(data) !== await core.digest(checked)) throw new Error('Backup self-check failed.');
    const url = URL.createObjectURL(new Blob([text], {type: 'application/octet-stream'}));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = crypto.randomUUID() + '.otbrowser';
    anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 10000);
    prepared = data; $('verify').disabled = false;
    $('backup-status').textContent = 'Save the download, then reopen it below and verify its passphrase.';
  });
  bind('verify', async () => {
    verified = null; $('delete').disabled = true;
    let password = $('verify-password').value; $('verify-password').value = '';
    const data = await core.unseal(await fileText('verify-file'), password); password = '';
    if (!prepared || await core.digest(data) !== await core.digest(prepared)) throw new Error('This is not the backup for the current selection.');
    verified = await core.digest(prepared); $('delete').disabled = false;
    $('backup-status').textContent = `Saved backup verified. Type DELETE ${prepared.rows.length} to apply.`;
    say('Verification succeeded. Deletion still requires the exact confirmation text.');
  });
  bind('delete', async () => {
    if (!prepared || !verified) throw new Error('Verify the saved backup first.');
    const data = prepared, confirmation = $('delete-confirm').value;
    if (confirmation !== `DELETE ${data.rows.length}`) throw new Error(`Type DELETE ${data.rows.length} exactly.`);
    $('stop').disabled = false;
    const report = await core.remove(api, data, {confirmation, backupDigest: verified, profile, engine}, showResults, () => cancelRequested);
    reset(); showResults(report); say(`${report.length} of ${data.rows.length} records processed. Review each result; refresh to see current data.`);
  });
  bind('inspect', async () => {
    restoration = null; $('restore').disabled = true; $('restore-summary').textContent = '';
    let password = $('restore-password').value; $('restore-password').value = '';
    const data = await core.unseal(await fileText('restore-file'), password); password = '';
    if (data.profile !== profile || data.engine !== engine) throw new Error('This backup belongs to another browser profile. Open it in the original profile.');
    restoration = data; $('restore').disabled = false;
    const lines = data.rows.map(row => data.kind === 'history' ? row.url : `${row.domain} | ${row.name} | ${row.path} | store ${row.storeId}`);
    $('restore-summary').textContent = `${data.kind}: ${data.rows.length} records. Type RESTORE ${data.rows.length}.\n` + lines.join('\n');
  });
  bind('restore', async () => {
    if (!restoration) throw new Error('Inspect a backup first.');
    const data = restoration; $('stop').disabled = false;
    const report = await core.restore(api, data, {confirmation: $('restore-confirm').value, profile, engine}, showResults, () => cancelRequested);
    restoration = null; $('restore').disabled = true; $('restore-confirm').value = ''; showResults(report);
    say(`Restore finished: ${report.length} records processed. History times are approximate; inspect individual results.`);
  });
  render();
})();
