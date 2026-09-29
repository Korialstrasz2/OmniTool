'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const csv = require('../static/csv.js');
const {Sheet} = require('../static/csv_model.js');
test('quoted multiline fields, escaped quotes, BOM and roundtrip', () => {
  const text = '\ufeffname,value\r\n"two\nlines","a ""quote"""\r\n';
  const rows = csv.parse(text);
  assert.deepEqual(rows, [['name','value'],['two\nlines','a "quote"']]);
  assert.deepEqual(csv.parse(csv.stringify(rows, ',', '"', false)), rows);
});
test('literal filtering retains original row indexes', () => {
  const sheet = new Sheet([['H'],['keep'],['other'],['KEEP again']]);
  assert.deepEqual(sheet.view({query:'keep'}).rows.map(r=>r.index), [1,3]);
  sheet.edit(3,0,'changed');
  assert.deepEqual(sheet.view({query:'keep'}).rows.map(r=>r.index), [1]);
  assert.equal(sheet.rows[2][0], 'other');
});
test('leading zeroes, formulas and numbers remain text', () => {
  const sheet = new Sheet(csv.parse('001,-2,=1+1'));
  assert.deepEqual(sheet.rows[0], ['001','-2','=1+1']);
  assert.ok(csv.stringify(sheet.rows).includes("'-2"));
  assert.equal(csv.stringify(sheet.rows, ',', '"', false), '001,-2,=1+1');
});
test('edits preserve ragged records and undo exact shape', () => {
  const sheet = new Sheet([['a','b'],['one']]);
  sheet.edit(1,1,'two'); assert.deepEqual(sheet.rows[1],['one','two']);
  sheet.undo(); assert.deepEqual(sheet.rows[1],['one']);
});
test('transforms preserve enabled header and can be undone', () => {
  const sheet = new Sheet([[' HEADER '],[' b '],[' a '],[' a '],[' ']]);
  sheet.transform('trim'); assert.equal(sheet.rows[0][0], ' HEADER ');
  sheet.transform('remove-empty'); sheet.transform('deduplicate'); sheet.transform('sort-asc');
  assert.deepEqual(sheet.rows, [[' HEADER '],['a'],['b']]);
  sheet.undo(); assert.deepEqual(sheet.rows, [[' HEADER '],['b'],['a']]);
  sheet.reset(); assert.equal(sheet.rows.length,5);
});
test('no-header mode transforms every row', () => {
  const sheet = new Sheet([[' b '],[' a ']]); sheet.transform('trim',0,false); sheet.transform('sort-asc',0,false);
  assert.deepEqual(sheet.rows,[['a'],['b']]);
});
test('row and column pagination bound rendered cells', () => {
  const sheet = new Sheet(Array.from({length:1001},(_,i)=>Array.from({length:25},(_,j)=>`${i}:${j}`)));
  const view = sheet.view({page:2,colPage:2});
  assert.equal(view.rows.length,50); assert.equal(view.rows[0].index,51); assert.equal(view.columns.length,5);
  assert.equal(view.rows[0].cells[0],'51:20');
});
test('filtered export is opt-in and includes header once', () => {
  const sheet = new Sheet([['a','a'],['yes','1'],['no','2']]);
  assert.equal(sheet.exportRows('yes',true,false).length,3);
  assert.deepEqual(sheet.exportRows('yes',true,true),[['a','a'],['yes','1']]);
});
test('literal filter does not execute regular expressions', () => {
  const sheet = new Sheet([['h'],['.*'],['other']]);
  assert.deepEqual(sheet.indexes('.*'),[1]);
});
test('malformed quotes and ambiguous unquoted export fail', () => {
  assert.throws(()=>csv.parse('"missing')); assert.throws(()=>csv.parse('"a"bad'));
  assert.throws(()=>csv.stringify([['a,b']],',','',false));
  assert.throws(()=>csv.stringify([['a']],',','xx',false));
});
test('input bounds and invalid edits fail without mutation', () => {
  assert.throws(()=>new Sheet([Array(2001).fill('')]));
  const sheet = new Sheet([['a']]);
  for (const args of [[-1,0,'x'],[0,-1,'x'],[0,0,1],[0,0,'x'.repeat(65537)]]) assert.throws(()=>sheet.edit(...args));
  assert.deepEqual(sheet.rows,[['a']]);
});
test('undo history is bounded, reset itself undoable', () => {
  const sheet = new Sheet([['a']]);
  for(let i=0;i<30;i++)sheet.edit(0,0,String(i));
  assert.equal(sheet.history.length,20); sheet.reset(); assert.equal(sheet.rows[0][0],'a');
  sheet.undo(); assert.equal(sheet.rows[0][0],'29');
});
test('empty and trailing records survive parser roundtrip', () => {
  assert.deepEqual(csv.parse('a,b,\n\n'),[['a','b',''],['']]);
  assert.deepEqual(csv.parse(''),[]);
});
test('auto detection ignores quoted delimiters', () => {
  assert.equal(csv.detect('a;b\n"x;y";z'),';');
  assert.equal(csv.detect('a\tb\n1\t2'),'\t');
});
test('append row and all-row empty removal', () => {
  const sheet = new Sheet([['a','b']]); sheet.transform('add-row');
  assert.deepEqual(sheet.rows,[['a','b'],['','']]); sheet.transform('remove-empty'); assert.equal(sheet.rows.length,1);
});
