'use strict';
window.OmniAPI = async function (path, data) {
  const options = {credentials: 'same-origin', cache: 'no-store', headers: {}};
  if (data !== undefined) {
    options.method = 'POST';
    options.headers['Content-Type'] = 'application/json';
    options.headers['X-CSRF-Token'] = document.querySelector('meta[name="csrf-token"]').content;
    options.body = JSON.stringify(data);
  }
  const response = await fetch(path, options);
  if (response.status === 401) { window.location.assign('/login'); throw new Error('Session expired'); }
  const json = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(json.error || `Request failed (${response.status})`);
  return json;
};
// A page restored from the back/forward cache must revalidate authentication and locks.
window.addEventListener('pageshow', event => { if (event.persisted) window.location.reload(); });
