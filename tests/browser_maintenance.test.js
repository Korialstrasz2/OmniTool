'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const core = require('../extensions/browser-maintenance/core.js');
const cookie = (changes = {}) => ({domain: '.example.co.uk', hostOnly: false, name: 'session', value: 'synthetic-test-only',
  path: '/', secure: true, httpOnly: true, sameSite: 'lax', session: true, storeId: '0', ...changes});
const data = (kind, rows) => ({version: 1, id: 'fixture', profile: 'test-profile', engine: 'chromium', created: 1, kind, rows});
const approval = async (snapshot) => ({profile: snapshot.profile, engine: snapshot.engine,
  backupDigest: await core.digest(snapshot), confirmation: `DELETE ${snapshot.rows.length}`});

test('hostname matching is exact and boundary-aware', () => {
  assert.equal(core.host('EXAMPLE.co.uk.'), 'example.co.uk');
  assert.equal(core.host('bücher.example'), 'xn--bcher-kva.example');
  assert(core.matches('.example.co.uk', 'example.co.uk'));
  assert(!core.matches('shop.example.co.uk', 'example.co.uk'));
  assert(core.matches('shop.example.co.uk', 'example.co.uk', true));
  assert(!core.matches('evil-example.co.uk', 'example.co.uk', true));
  assert(!core.matches('example.co.uk.evil.test', 'example.co.uk', true));
  for (const value of ['', '*.example.com', 'https://example.com', 'example.com/path', 'x@y.com', 'x%2ey.com']) assert.throws(() => core.host(value));
});
test('history previews ignore matching text in paths, queries, titles, and credentials', async () => {
  const api = {history: {search: async () => [
    {url: 'https://example.co.uk/ok'}, {url: 'https://sub.example.co.uk/ok'},
    {url: 'https://evil.test/?q=example.co.uk'}, {url: 'https://example.co.uk@evil.test/'},
    {url: 'https://example.co.uk.evil.test/'}, {url: 'file:///example.co.uk'},
  ]}};
  assert.equal((await core.historyPreview(api, 'example.co.uk', false)).rows.length, 1);
  assert.equal((await core.historyPreview(api, 'example.co.uk', true)).rows.length, 2);
});
test('cookie allowlist does not collapse multi-label or hosted-service domains', async () => {
  let query;
  const api = {cookies: {getAll: async q => {query=q; return [cookie(), cookie({domain:'.another.co.uk'}), cookie({domain:'alice.github.io'}), cookie({domain:'bob.github.io'})];}}};
  const result = await core.cookiePreview(api, 'container-1', ['example.co.uk','alice.github.io'], false, true);
  assert.equal(result.protectedCount, 2);
  assert.deepEqual(query, {storeId:'container-1', partitionKey:{}, firstPartyDomain:null});
});
test('cookie identity includes path, store, partition, and first-party isolation', () => {
  const base = cookie();
  for (const change of [{path:'/other'}, {storeId:'1'}, {domain:'example.co.uk',hostOnly:true},
    {partitionKey:{topLevelSite:'https://site.test',hasCrossSiteAncestor:true}}, {firstPartyDomain:'site.test'}]) {
    assert.notEqual(core.cookieKey(base), core.cookieKey(cookie(change)));
  }
});
test('unaddressable cookie paths fail closed', () => {
  for (const path of ['/a/../b', '/?x=1', '/#fragment', '/a b']) assert.throws(() => core.cookieDetails(cookie({path})));
});
test('partition and isolation selectors reach get/remove unchanged', async () => {
  const item = cookie({partitionKey:{topLevelSite:'https://site.test',hasCrossSiteAncestor:true},firstPartyDomain:'site.test'});
  const snapshot = data('cookies',[item]); let selector;
  const api = {cookies:{get:async()=>item, remove:async d=>{selector=d;return d;},getAll:async()=>[]}};
  const report=await core.remove(api,snapshot,await approval(snapshot));
  assert.equal(report[0].status,'removed'); assert.equal(selector.storeId,'0');
  assert.deepEqual(selector.partitionKey,item.partitionKey); assert.equal(selector.firstPartyDomain,'site.test');
});
test('stale or shadowed cookies are skipped without deletion', async () => {
  const item=cookie(), snapshot=data('cookies',[item]); let calls=0;
  const api={cookies:{get:async()=>cookie({path:'/different'}),remove:async()=>{calls++;}}};
  assert.equal((await core.remove(api,snapshot,await approval(snapshot)))[0].status,'skipped-changed-or-shadowed');
  assert.equal(calls,0);
  api.cookies.get=async()=>cookie({value:'changed'});
  await core.remove(api,snapshot,await approval(snapshot)); assert.equal(calls,0);
});
test('stale history visits are skipped without deleting the URL', async () => {
  const visits=[{visitId:'1',visitTime:1,transition:'link',referringVisitId:'0'}];
  const snapshot=data('history',[{url:'https://example.test/',visits}]);let calls=0;
  const api={history:{getVisits:async()=>[...visits,{visitId:'2',visitTime:2}],deleteUrl:async()=>{calls++;}}};
  assert.equal((await core.remove(api,snapshot,await approval(snapshot)))[0].status,'skipped-changed');assert.equal(calls,0);
});
test('approval rejects wrong profile, backup digest, and confirmation', async () => {
  const snapshot=data('cookies',[cookie()]); const good=await approval(snapshot);
  for(const change of [{profile:'other'},{backupDigest:'bad'},{confirmation:'DELETE'},{engine:'firefox'}]) {
    await assert.rejects(core.remove({},snapshot,{...good,...change}));
  }
});
test('cancellation stops before the next record and errors are itemized', async () => {
  const snapshot=data('cookies',[cookie(),cookie({name:'other'})]); let count=0;
  const api={cookies:{get:async()=>{throw Error('never expose sensitive details');}}};
  const report=await core.remove(api,snapshot,await approval(snapshot),()=>{count++;},()=>count===1);
  assert.deepEqual(report,[{index:0,status:'failed'}]);
});
test('restore skips existing and expired cookies; never calls set', async () => {
  let calls=0;const existing=cookie(), expired=cookie({name:'expired',session:false,expirationDate:1});
  const snapshot=data('cookies',[existing,expired]);
  const api={cookies:{getAll:async()=>[existing],set:async()=>{calls++;}}};
  const report=await core.restore(api,snapshot,{profile:'test-profile',engine:'chromium',confirmation:'RESTORE 2'});
  assert.deepEqual(report.map(r=>r.status),['skipped-existing','skipped-expired']);assert.equal(calls,0);
});
test('history restore is explicitly URL-only at the current time', async () => {
  let added;const snapshot=data('history',[{url:'https://example.test/',visits:[{visitTime:1}]}]);
  const api={history:{getVisits:async()=>[],addUrl:async details=>{added=details;}}};
  const report=await core.restore(api,snapshot,{profile:'test-profile',engine:'chromium',confirmation:'RESTORE 1'});
  assert.deepEqual(added,{url:'https://example.test/'});assert.equal(report[0].status,'url-restored-at-current-time');
});
test('encrypted backups round-trip without plaintext; wrong passwords and tampering fail', async () => {
  const snapshot=data('cookies',[cookie()]), password='test-only-generated-passphrase';
  const sealed=await core.seal(snapshot,password);
  assert(!sealed.includes('synthetic-test-only'));assert(!sealed.includes('example.co.uk'));
  assert.deepEqual(await core.unseal(sealed,password),snapshot);
  assert.notEqual(sealed,await core.seal(snapshot,password));
  await assert.rejects(core.unseal(sealed,password+'wrong'));
  const box=JSON.parse(sealed);box.data=(box.data[0]==='A'?'B':'A')+box.data.slice(1);
  await assert.rejects(core.unseal(JSON.stringify(box),password));
  await assert.rejects(core.seal(snapshot,'weak'));
});
test('backup validation rejects duplicates and unsupported partition fields', () => {
  assert.throws(()=>core.validateBackup(data('cookies',[cookie(),cookie()])));
  assert.throws(()=>core.validateBackup(data('cookies',[cookie({partitionKey:{topLevelSite:'https://x.test',unknown:1}})])));
  assert.throws(()=>core.validateBackup(data('history',[{url:'javascript:alert(1)',visits:[]}])));
});
