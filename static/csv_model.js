'use strict';
const OmniSheet = (() => {
  const MAX_CELLS = 200000, MAX_TEXT = 5 * 1024 * 1024, MAX_UNDO = 20 * 1024 * 1024;
  function validate(rows) {
    if (!Array.isArray(rows) || rows.length > 50000) throw new Error('At most 50,000 rows are supported');
    let cells = 0, chars = 0;
    for (const row of rows) {
      if (!Array.isArray(row) || row.length > 2000) throw new Error('At most 2,000 columns are supported');
      cells += row.length;
      for (const value of row) {
        if (typeof value !== 'string') throw new Error('Cell values must remain text');
        chars += value.length;
      }
    }
    if (cells > MAX_CELLS || chars > MAX_TEXT) throw new Error('Table exceeds its cell/text limit');
  }
  class Sheet {
    constructor(rows) {
      validate(rows); this.original = JSON.stringify(rows); this.rows = rows.map(r => r.slice());
      this.history = []; this.bytes = 0;
    }
    commit(next) {
      validate(next);
      const before = JSON.stringify(this.rows);
      if (before === JSON.stringify(next)) return;
      this.history.push(before); this.bytes += before.length * 2;
      while (this.history.length > 20 || this.bytes > MAX_UNDO) this.bytes -= this.history.shift().length * 2;
      this.rows = next;
    }
    edit(row, column, value) {
      if (!Number.isInteger(row) || row < 0 || row >= this.rows.length || !Number.isInteger(column) ||
          column < 0 || column >= this.width() || typeof value !== 'string' || value.length > 65536)
        throw new Error('Invalid cell or value; edits are limited to 65,536 characters');
      const next = this.rows.map(r => r.slice());
      while (next[row].length <= column) next[row].push('');
      next[row][column] = value; this.commit(next);
    }
    width() { return this.rows.reduce((n, row) => Math.max(n, row.length), 0); }
    undo() {
      if (!this.history.length) return;
      const before = this.history.pop(); this.bytes -= before.length * 2; this.rows = JSON.parse(before);
    }
    reset() { this.commit(JSON.parse(this.original)); }
    transform(kind, column = 0, header = true) {
      if (typeof header !== 'boolean') throw new Error('Invalid header setting');
      let rows = this.rows.map(r => r.slice());
      const prefix = header && rows.length ? [rows.shift()] : [];
      if (kind === 'trim') rows = rows.map(r => r.map(v => v.trim()));
      else if (kind === 'remove-empty') rows = rows.filter(r => r.some(v => v.trim() !== ''));
      else if (kind === 'deduplicate') {
        const seen = new Set(); rows = rows.filter(r => { const key = JSON.stringify(r); if (seen.has(key)) return false; seen.add(key); return true; });
      } else if (kind === 'sort-asc' || kind === 'sort-desc') {
        if (!Number.isInteger(column) || column < 0 || column >= this.width()) throw new Error('Select a valid column');
        const sign = kind === 'sort-asc' ? 1 : -1;
        // Text order is deliberate: do not silently coerce account numbers or dates.
        rows.sort((a, b) => ((a[column] || '') < (b[column] || '') ? -1 : (a[column] || '') > (b[column] || '') ? 1 : 0) * sign);
      } else if (kind === 'add-row') rows.push(Array(Math.max(1, this.width())).fill(''));
      else throw new Error('Unknown transformation');
      this.commit([...prefix, ...rows]);
    }
    indexes(query = '', header = true) {
      if (typeof query !== 'string' || query.length > 256) throw new Error('Filter is too long');
      query = query.toLocaleLowerCase();
      return this.rows.map((_, i) => i).filter(i => !(header && i === 0) &&
        (!query || this.rows[i].some(v => v.toLocaleLowerCase().includes(query))));
    }
    view({query = '', header = true, page = 1, colPage = 1} = {}) {
      const indexes = this.indexes(query, header), pages = Math.max(1, Math.ceil(indexes.length / 50));
      const width = this.width(), colPages = Math.max(1, Math.ceil(width / 20));
      page = Math.min(pages, Math.max(1, Number.isInteger(page) ? page : 1));
      colPage = Math.min(colPages, Math.max(1, Number.isInteger(colPage) ? colPage : 1));
      const start = (colPage - 1) * 20;
      return {page, pages, colPage, colPages, width, count: this.rows.length, matches: indexes.length,
        undo: this.history.length, columns: Array.from({length: Math.min(20, Math.max(0, width - start))}, (_, j) => j + start),
        headings: header && this.rows.length ? this.rows[0].slice(start, start + 20) : null,
        rows: indexes.slice((page - 1) * 50, page * 50).map(index => ({index, cells: this.rows[index].slice(start, start + 20)})),
        ragged: this.rows.some(row => row.length !== width)};
    }
    exportRows(query = '', header = true, filtered = false) {
      const indexes = filtered ? this.indexes(query, header) : this.rows.map((_, i) => i).filter(i => !(header && i === 0));
      return [...(header && this.rows.length ? [this.rows[0]] : []), ...indexes.map(i => this.rows[i])];
    }
  }
  return {Sheet, validate};
})();
if (typeof module !== 'undefined') module.exports = OmniSheet;
if (typeof self !== 'undefined') self.OmniSheet = OmniSheet;
