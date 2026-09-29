'use strict';
// Pure parser/serializer, shared by the worker and the Node regression suite.
const OmniCSV = (() => {
  function separators(delimiter, quote) {
    if (typeof delimiter !== 'string' || delimiter.length !== 1 || /[\r\n\0]/.test(delimiter) ||
        !['', '"', "'"].includes(quote) || delimiter === quote) throw new Error('Invalid separators');
  }
  function parse(text, delimiter = ',', quote = '"') {
    separators(delimiter, quote);
    text = text.replace(/^\uFEFF/, '');
    if (text.includes('\0')) throw new Error('NUL characters found; check the selected text encoding');
    if (!text.length) return [];
    const rows = []; let row = [], field = '', quoted = false, closed = false, ended = false, cells = 0;
    function cell() { if (++cells > 200000) throw new Error('File exceeds the 200,000-cell limit'); row.push(field); field = ''; closed = false; }
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
      } catch (_) { /* Another separator may still be valid. */ }
    }
    return best;
  }
  function stringify(rows, delimiter = ',', quote = '"', safe = true) {
    separators(delimiter, quote);
    return rows.map(row => row.map(value => {
      let text = String(value ?? '');
      if (safe && (/^[\t\r\n]/.test(text) || /^[\t\r\n ]*[=+\-@]/.test(text))) text = "'" + text;
      const needsQuote = text.includes(delimiter) || /[\r\n]/.test(text) || (quote && text.includes(quote));
      if (needsQuote && !quote) throw new Error('This data needs quoting; choose a quote character');
      return needsQuote ? quote + text.split(quote).join(quote + quote) + quote : text;
    }).join(delimiter)).join('\r\n');
  }
  return {parse, detect, stringify};
})();
if (typeof module !== 'undefined') module.exports = OmniCSV;
if (typeof self !== 'undefined') self.OmniCSV = OmniCSV;
