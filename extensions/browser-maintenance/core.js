'use strict';
// Native browser APIs only. No SQLite, native messaging, fetch, or content scripts.
const OmniBrowserCore = (() => {
  const MAX_ROWS = 500, MAX_BYTES = 8 * 1024 * 1024;
  const encoder = new TextEncoder(), decoder = new TextDecoder('utf-8', {fatal: true});
  function stable(value) {
    if (Array.isArray(value)) return '[' + value.map(stable).join(',') + ']';
    if (value && typeof value === 'object') return '{' + Object.keys(value).sort().map(k => JSON.stringify(k) + ':' + stable(value[k])).join(',') + '}';
    return JSON.stringify(value);
  }
  function host(value) {
    if (typeof value !== 'string') throw new Error('Enter a hostname, not a URL.');
    const name = value.trim().replace(/\.$/, '');
    if (!name || /[\s\/@:#?%*\\]/u.test(name)) throw new Error('Enter a hostname only, such as example.co.uk.');
    const normalized = new URL('https://' + name).hostname.toLowerCase();
    if (normalized.length > 253 || !normalized.split('.').every(s => /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/.test(s))) throw new Error('Invalid hostname.');
    return normalized;
  }
  function matches(candidate, selected, subdomains = false) {
    try {
      const a = host(candidate.replace(/^\./, '')), b = host(selected);
      return a === b || (subdomains && a.endsWith('.' + b));
    } catch (_) { return false; }
  }
  function validURL(value) {
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Only HTTP(S) records are supported.');
    return url;
  }
  function cookieKey(c) {
    return stable([c.storeId, c.domain, c.hostOnly, c.path, c.name,
      c.firstPartyDomain ?? null, c.partitionKey ?? null]);
  }
  function cookieSignature(c) {
    return stable([cookieKey(c), c.value, c.secure, c.httpOnly, c.sameSite, c.session, c.expirationDate ?? null]);
  }
  function cookieDetails(c) {
    const hostname = host(c.domain.replace(/^\./, ''));
    if (typeof c.path !== 'string' || !c.path.startsWith('/')) throw new Error('Unsupported cookie path.');
    const text = `${c.secure ? 'https' : 'http'}://${hostname}${c.path}`;
    const parsed = validURL(text);
    // URL normalization must not silently point removal at a different path.
    if (parsed.hostname !== hostname || parsed.pathname !== c.path || parsed.search || parsed.hash) throw new Error('This cookie path cannot be addressed safely by the browser API.');
    const details = {url: text, name: c.name, storeId: c.storeId};
    if (c.partitionKey) details.partitionKey = {...c.partitionKey};
    if (c.firstPartyDomain !== undefined) details.firstPartyDomain = c.firstPartyDomain;
    return details;
  }
  function visitSignature(visits) {
    return stable(visits.map(v => [v.visitId, v.visitTime, v.transition, v.referringVisitId, v.isLocal ?? null]).sort((a, b) => stable(a).localeCompare(stable(b))));
  }
  async function historyPreview(api, domain, subdomains) {
    domain = host(domain);
    // Never use a substring match in a URL or title as a deletion predicate.
    const found = await api.history.search({text: '', startTime: 0, maxResults: 20000});
    const rows = found.filter(item => {
      try { return matches(validURL(item.url).hostname, domain, subdomains); } catch (_) { return false; }
    });
    if (rows.length > MAX_ROWS) throw new Error('More than 500 matching URLs. Narrow the hostname before proceeding.');
    return {rows, truncated: found.length >= 20000};
  }
  async function cookiesInStore(api, storeId, firefox) {
    if (!storeId) throw new Error('Select an explicit cookie store.');
    const details = {storeId, partitionKey: {}};
    if (firefox) details.firstPartyDomain = null;
    return api.cookies.getAll(details); // Errors are not retried without partition isolation.
  }
  async function cookiePreview(api, storeId, allow, subdomains, firefox) {
    const domains = allow.map(host);
    const all = await cookiesInStore(api, storeId, firefox);
    const rows = all.filter(c => !domains.some(d => matches(c.domain, d, subdomains)));
    if (rows.length > MAX_ROWS) throw new Error('More than 500 unprotected cookies. Extend the allowlist or use browser settings.');
    return {rows, protectedCount: all.length - rows.length};
  }
  function validateBackup(data) {
    if (!data || data.version !== 1 || !['history', 'cookies'].includes(data.kind) || !['chromium', 'firefox'].includes(data.engine)
        || typeof data.profile !== 'string' || typeof data.id !== 'string' || !Array.isArray(data.rows)
        || !data.rows.length || data.rows.length > MAX_ROWS) throw new Error('Invalid browser backup.');
    const seen = new Set();
    for (const row of data.rows) {
      let key;
      if (data.kind === 'history') {
        if (typeof row.url !== 'string' || row.url.length > 65536 || !Array.isArray(row.visits)) throw new Error('Invalid history record.');
        validURL(row.url); key = row.url;
      } else {
        if (['name', 'value', 'domain', 'path', 'storeId', 'sameSite'].some(k => typeof row[k] !== 'string')
            || ['secure', 'httpOnly', 'hostOnly', 'session'].some(k => typeof row[k] !== 'boolean')) throw new Error('Invalid cookie record.');
        if (!['unspecified', 'no_restriction', 'lax', 'strict'].includes(row.sameSite)) throw new Error('Invalid SameSite value.');
        if (row.expirationDate !== undefined && (!Number.isFinite(row.expirationDate) || row.expirationDate < 0)) throw new Error('Invalid cookie expiry.');
        if (row.firstPartyDomain !== undefined && typeof row.firstPartyDomain !== 'string') throw new Error('Invalid first-party domain.');
        if (row.partitionKey !== undefined) {
          if (!row.partitionKey || typeof row.partitionKey.topLevelSite !== 'string'
              || Object.keys(row.partitionKey).some(k => !['topLevelSite', 'hasCrossSiteAncestor'].includes(k))) throw new Error('Unsupported partition key.');
          validURL(row.partitionKey.topLevelSite);
          if (row.partitionKey.hasCrossSiteAncestor !== undefined && typeof row.partitionKey.hasCrossSiteAncestor !== 'boolean') throw new Error('Invalid partition key.');
        }
        cookieDetails(row); key = cookieKey(row);
      }
      if (seen.has(key)) throw new Error('Duplicate backup record.');
      seen.add(key);
    }
    if (encoder.encode(JSON.stringify(data)).length > MAX_BYTES) throw new Error('Backup exceeds 8 MiB.');
    return data;
  }
  async function snapshot(api, kind, selected, profile, engine) {
    if (!selected.length || selected.length > MAX_ROWS) throw new Error('Select between 1 and 500 records.');
    const rows = [];
    for (const row of selected) {
      if (kind === 'history') {
        const visits = await api.history.getVisits({url: row.url});
        if (!visits.length) throw new Error('History changed. Create a fresh preview.');
        rows.push({url: row.url, title: row.title || '', visits});
      } else {
        const current = await api.cookies.get(cookieDetails(row));
        if (!current || cookieSignature(current) !== cookieSignature(row)) throw new Error('A selected cookie changed or is shadowed by another cookie. Refresh and select it separately.');
        rows.push({...row});
      }
    }
    return validateBackup({version: 1, id: crypto.randomUUID(), kind, profile, engine, created: Date.now(), rows});
  }
  async function digest(data) {
    return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', encoder.encode(stable(data)))), x => x.toString(16).padStart(2, '0')).join('');
  }
  async function remove(api, data, approval, progress = () => {}, canceled = () => false) {
    validateBackup(data);
    if (approval.confirmation !== `DELETE ${data.rows.length}` || approval.backupDigest !== await digest(data)
        || approval.profile !== data.profile || approval.engine !== data.engine) throw new Error('Verify the saved backup and confirm the exact selection first.');
    const report = [];
    for (const [index, row] of data.rows.entries()) {
      if (canceled()) break;
      let status;
      try {
        if (data.kind === 'history') {
          const current = await api.history.getVisits({url: row.url});
          if (visitSignature(current) !== visitSignature(row.visits)) status = 'skipped-changed';
          else {
            await api.history.deleteUrl({url: row.url});
            status = (await api.history.getVisits({url: row.url})).length ? 'new-visits-remain' : 'removed';
          }
        } else {
          const details = cookieDetails(row), current = await api.cookies.get(details);
          if (!current || cookieSignature(current) !== cookieSignature(row)) status = 'skipped-changed-or-shadowed';
          else {
            const result = await api.cookies.remove(details);
            const remaining = await cookiesInStore(api, row.storeId, data.engine === 'firefox');
            status = !result ? 'already-missing' : (remaining.some(c => cookieKey(c) === cookieKey(row)) ? 'cookie-recreated' : 'removed');
          }
        }
      } catch (_) { status = 'failed'; } // Never log cookie values or sensitive URL strings.
      report.push({index, status}); progress(report.slice());
    }
    return report;
  }
  async function restore(api, data, approval, progress = () => {}, canceled = () => false) {
    validateBackup(data);
    if (approval.confirmation !== `RESTORE ${data.rows.length}` || approval.profile !== data.profile || approval.engine !== data.engine) throw new Error('Restore only in the original browser profile after explicit confirmation.');
    const report = [];
    for (const [index, row] of data.rows.entries()) {
      if (canceled()) break;
      let status;
      try {
        if (data.kind === 'history') {
          if ((await api.history.getVisits({url: row.url})).length) status = 'skipped-existing';
          else { await api.history.addUrl({url: row.url}); status = 'url-restored-at-current-time'; }
        } else if (row.expirationDate !== undefined && row.expirationDate <= Date.now() / 1000) status = 'skipped-expired';
        else {
          const all = await cookiesInStore(api, row.storeId, data.engine === 'firefox');
          if (all.some(c => cookieKey(c) === cookieKey(row))) status = 'skipped-existing';
          else {
            const details = {...cookieDetails(row), value: row.value, path: row.path, secure: row.secure,
              httpOnly: row.httpOnly, sameSite: row.sameSite};
            if (!row.hostOnly) details.domain = row.domain;
            if (!row.session && row.expirationDate !== undefined) details.expirationDate = row.expirationDate;
            await api.cookies.set(details);
            const check = await cookiesInStore(api, row.storeId, data.engine === 'firefox');
            status = check.some(c => cookieSignature(c) === cookieSignature(row)) ? 'restored' : 'restore-not-exact';
          }
        }
      } catch (_) { status = 'failed'; }
      report.push({index, status}); progress(report.slice());
    }
    return report;
  }
  const aad = encoder.encode('OmniTool browser backup v1');
  function b64(bytes) {
    let text = '';
    for (let i = 0; i < bytes.length; i += 8192) text += String.fromCharCode(...bytes.subarray(i, i + 8192));
    return btoa(text);
  }
  function unb64(text) {
    if (typeof text !== 'string' || text.length > MAX_BYTES * 1.4 + 1024) throw new Error('Invalid encrypted backup.');
    return Uint8Array.from(atob(text), c => c.charCodeAt(0));
  }
  async function key(password, salt) {
    if (typeof password !== 'string' || password.length < 16 || password.length > 1024) throw new Error('Use a generated passphrase of 16–1024 characters.');
    const material = await crypto.subtle.importKey('raw', encoder.encode(password), 'PBKDF2', false, ['deriveKey']);
    return crypto.subtle.deriveKey({name: 'PBKDF2', salt, iterations: 600000, hash: 'SHA-256'}, material,
      {name: 'AES-GCM', length: 256}, false, ['encrypt', 'decrypt']);
  }
  async function seal(data, password) {
    validateBackup(data);
    const salt = crypto.getRandomValues(new Uint8Array(16)), iv = crypto.getRandomValues(new Uint8Array(12));
    const ciphertext = await crypto.subtle.encrypt({name: 'AES-GCM', iv, additionalData: aad}, await key(password, salt), encoder.encode(JSON.stringify(data)));
    return JSON.stringify({format: 'omnitool-browser-backup', version: 1, salt: b64(salt), iv: b64(iv), data: b64(new Uint8Array(ciphertext))});
  }
  async function unseal(text, password) {
    if (text.length > MAX_BYTES * 1.4 + 2048) throw new Error('Encrypted backup exceeds its size limit.');
    const box = JSON.parse(text);
    if (box.format !== 'omnitool-browser-backup' || box.version !== 1) throw new Error('Unsupported backup format.');
    const salt = unb64(box.salt), iv = unb64(box.iv), data = unb64(box.data);
    if (salt.length !== 16 || iv.length !== 12 || data.length > MAX_BYTES + 16) throw new Error('Invalid encrypted backup.');
    let plain;
    try { plain = await crypto.subtle.decrypt({name: 'AES-GCM', iv, additionalData: aad}, await key(password, salt), data); }
    catch (_) { throw new Error('Incorrect passphrase or damaged backup.'); }
    return validateBackup(JSON.parse(decoder.decode(plain)));
  }
  return {host, matches, cookieKey, cookieSignature, cookieDetails, visitSignature, historyPreview,
    cookiePreview, cookiesInStore, validateBackup, snapshot, digest, remove, restore, seal, unseal};
})();
if (typeof module !== 'undefined') module.exports = OmniBrowserCore;
