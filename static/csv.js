'use strict';
const OmniCSV = (() => {
  function parse(text, delimiter = ',', quote = '"') {
    if (delimiter.length !== 1 || (quote && quote.length !== 1) || delimiter === quote) throw new Error('Invalid separators');
    text = text.replace(/^\uFEFF/, '');
    if (!text.length) return [];
    const rows = []; let row = [], field = '', quoted = false, closed = false, ended = false, cells = 0;
    function cell() { if (++cells > 1000000) throw new Error('File exceeds the one-million-cell limit'); row.push(field); field = ''; closed = false; }
    function record() { cell(); rows.push(row); row = []; ended = true; }
    for (let i = 0; i < text.length; i++) {
      const c = text[i]; ended = false;
      if (quoted) {
        if (c === quote) { if (text[i + 1] === quote) { field += quote; i++; } else { quoted = false; closed = true; } }
        else field += c;
      } else if (c === delimiter) cell();
      else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; record(); }
      else if (closed) throw new Error('Unexpected text after a closing quote');
      else if (quote && c === quote && field === '') quoted = true;
      else if (quote && c === quote) throw new Error('Unexpected quote in an unquoted field');
      else field += c;
    }
    if (quoted) throw new Error('Unclosed quoted field');
    if (!ended) record();
    return rows;
  }
  function detect(text, quote = '"') {
    let best = ',', high = -Infinity;
    for (const delimiter of [',', ';', '\t', '|', ':']) {
      if (delimiter === quote) continue;
      try {
        const rows = parse(text, delimiter, quote).slice(0, 50);
        const counts = new Map();
        for (const row of rows) counts.set(row.length, (counts.get(row.length) || 0) + 1);
        let score = 0;
        for (const [width, count] of counts) if (width > 1) score = Math.max(score, count / Math.max(1, rows.length) * 100 + Math.min(width, 20));
        if (score > high) { high = score; best = delimiter; }
      } catch (_) { /* Other separators may still be valid. */ }
    }
    return best;
  }
  function stringify(rows, delimiter = ',', quote = '"', safe = true) {
    if (delimiter.length !== 1 || delimiter === quote) throw new Error('Invalid output separator');
    return rows.map(row => row.map(value => {
      let text = String(value ?? '');
      if (safe && /^[\t\r\n ]*[=+\-@]/.test(text)) text = "'" + text;
      const needsQuote = text.includes(delimiter) || /[\r\n]/.test(text) || (quote && text.includes(quote));
      if (needsQuote && !quote) throw new Error('This data needs quoting; choose a quote character');
      if (needsQuote) return quote + text.split(quote).join(quote + quote) + quote;
      return text;
    }).join(delimiter)).join('\r\n');
  }
  return {parse, detect, stringify};
})();
if (typeof module !== 'undefined') module.exports = OmniCSV;
if (typeof document !== 'undefined') {
  const file = document.getElementById('csv-file');
  if (file) {
    let raw = '', rows = [], filename = 'converted.csv';
    const delimiter = document.getElementById('import-delimiter');
    const quote = document.getElementById('import-quote');
    const status = document.getElementById('csv-status');
    const output = document.getElementById('csv-export');
    function preview() {
      const table = document.getElementById('preview-table'); table.replaceChildren(); output.disabled = true;
      try {
        const separator = delimiter.value === 'auto' ? OmniCSV.detect(raw, quote.value) : delimiter.value;
        rows = OmniCSV.parse(raw, separator, quote.value);
        for (const [i, row] of rows.slice(0, 40).entries()) {
          const tr = document.createElement('tr');
          for (const value of row.slice(0, 40)) { const td = document.createElement(i === 0 ? 'th' : 'td'); td.textContent = value; tr.append(td); }
          table.append(tr);
        }
        status.textContent = `${rows.length} rows · separator ${separator === '\t' ? 'tab' : separator} · preview limited to 40 × 40 cells`;
        output.disabled = !rows.length;
      } catch (error) { rows = []; status.textContent = error.message; }
    }
    file.addEventListener('change', async () => {
      const selected = file.files[0]; if (!selected) return;
      if (selected.size > 5 * 1024 * 1024) { status.textContent = 'Choose a file smaller than 5 MiB. Large-file streaming is not implemented yet.'; output.disabled = true; return; }
      raw = await selected.text(); filename = selected.name.replace(/\.[^.]*$/, '') + '-converted.csv'; preview();
    });
    [delimiter, quote].forEach(control => control.addEventListener('change', preview));
    output.addEventListener('click', () => {
      try {
        const text = OmniCSV.stringify(rows, document.getElementById('export-delimiter').value,
          document.getElementById('export-quote').value, document.getElementById('safe-formulas').checked);
        const url = URL.createObjectURL(new Blob([text], {type: 'text/csv;charset=utf-8'}));
        const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      } catch (error) { status.textContent = error.message; }
    });
  }
}
