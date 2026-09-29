'use strict';
importScripts('csv.js', 'csv_model.js');
let sheet = null, delimiter = ',', chain = Promise.resolve();
// Serialize operations, including asynchronous file reads: imports cannot race edits.
self.onmessage = event => {
  const request = event.data;
  chain = chain.then(async () => {
    const {id, action, data = {}, view = {}} = request;
    try {
      if (action === 'load') {
        sheet = null;
        if (!data.file || data.file.size > 5 * 1024 * 1024) throw new Error('Select a file up to 5 MiB');
        if (!['utf-8', 'utf-16le', 'windows-1252'].includes(data.encoding)) throw new Error('Unsupported encoding');
        const text = new TextDecoder(data.encoding, {fatal: true}).decode(await data.file.arrayBuffer());
        delimiter = data.delimiter === 'auto' ? OmniCSV.detect(text, data.quote) : data.delimiter;
        sheet = new OmniSheet.Sheet(OmniCSV.parse(text, delimiter, data.quote));
      } else {
        if (!sheet) throw new Error('Import a CSV file first');
        if (action === 'edit') sheet.edit(data.row, data.column, data.value);
        else if (action === 'transform') sheet.transform(data.kind, data.column, view.header);
        else if (action === 'undo') sheet.undo();
        else if (action === 'reset') sheet.reset();
        else if (action === 'export') {
          const rows = sheet.exportRows(view.query, view.header, data.filtered);
          const text = data.format === 'json' ? JSON.stringify(rows, null, 2) : OmniCSV.stringify(rows, data.delimiter, data.quote, data.safe);
          self.postMessage({id, result: {text, format: data.format, rows: rows.length}}); return;
        } else if (action !== 'view') throw new Error('Unsupported operation');
      }
      self.postMessage({id, result: {...sheet.view(view), delimiter}});
    } catch (error) { self.postMessage({id, error: error.message}); }
  });
};
