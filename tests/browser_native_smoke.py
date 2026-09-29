"""Real Chromium APIs in a NEW DISPOSABLE profile; never attach to a user's browser.

pip install playwright; playwright install chromium
python tests/browser_native_smoke.py
Test-only manifest grants optional permissions up front and adds bookmarks solely
for preservation assertions. Runtime application JS is copied without modification.
"""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

def main():
    with tempfile.TemporaryDirectory(prefix='omnitool-browser-fixture-') as directory:
        temp = Path(directory)
        extension = temp / 'extension'
        shutil.copytree(ROOT / 'extensions/browser-maintenance', extension)
        manifest = json.loads((extension / 'manifest.json').read_text())
        manifest['permissions'] += manifest.pop('optional_permissions') + ['bookmarks']
        manifest['host_permissions'] = manifest.pop('optional_host_permissions')
        (extension / 'manifest.json').write_text(json.dumps(manifest))
        extension_id = ''.join(chr(97 + int(c, 16)) for c in hashlib.sha256(str(extension).encode()).hexdigest()[:32])
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(str(temp / 'profile'), headless=True,
                executable_path=shutil.which('chromium') or None, ignore_default_args=['--disable-extensions'],
                args=[f'--disable-extensions-except={extension}', f'--load-extension={extension}',
                      '--disable-background-networking', '--disable-component-update', '--no-first-run'])
            try:
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'chrome-extension://{extension_id}/workspace.html')
                page.wait_for_function("document.querySelector('#page-label').textContent.includes('Page')")
                result = page.evaluate('''async () => {
                  const c = OmniBrowserCore, api = chrome;
                  const target = 'https://repair-fixture.invalid/target';
                  const unrelated = 'https://unrelated.invalid/?q=repair-fixture.invalid';
                  await api.history.addUrl({url: target}); await api.history.addUrl({url: unrelated});
                  const mark = await api.bookmarks.create({title:'disposable fixture',url:target});
                  const found = await c.historyPreview(api,'repair-fixture.invalid',false);
                  if (found.rows.length !== 1) throw Error('hostname filter failed');
                  const snapshot = await c.snapshot(api,'history',found.rows,'fixture','chromium');
                  const backup = await c.seal(snapshot,'generated-test-only-passphrase');
                  const loaded = await c.unseal(backup,'generated-test-only-passphrase');
                  const results = await c.remove(api,loaded,{confirmation:'DELETE 1',backupDigest:await c.digest(loaded),profile:'fixture',engine:'chromium'});
                  if(results[0].status!=='removed') throw Error('history removal failed: '+JSON.stringify(results));
                  if(!(await api.history.getVisits({url:unrelated})).length) throw Error('unrelated history removed');
                  if((await api.bookmarks.get(mark.id))[0].url!==target) throw Error('bookmark changed');
                  const restored = await c.restore(api,loaded,{confirmation:'RESTORE 1',profile:'fixture',engine:'chromium'});
                  if(restored[0].status!=='url-restored-at-current-time') throw Error('URL restore failed');
                  const store = (await api.cookies.getAllCookieStores())[0].id;
                  await api.cookies.set({url:'https://repair-fixture.invalid/',name:'same-name',value:'root-fixture',path:'/',secure:true,storeId:store});
                  await api.cookies.set({url:'https://repair-fixture.invalid/nested',name:'same-name',value:'path-fixture',path:'/nested',secure:true,storeId:store});
                  await api.cookies.set({url:'https://repair-fixture.invalid/',name:'partition-fixture',value:'partition-fixture',path:'/',secure:true,storeId:store,partitionKey:{topLevelSite:'https://top.invalid',hasCrossSiteAncestor:true}});
                  const cookies = await c.cookiesInStore(api,store,false);
                  const chosen = cookies.filter(x=>x.path==='/nested'||x.name==='partition-fixture');
                  if(chosen.length!==2) throw Error('partitioned enumeration failed');
                  const cs = await c.snapshot(api,'cookies',chosen,'fixture','chromium');
                  const removed = await c.remove(api,cs,{confirmation:'DELETE 2',backupDigest:await c.digest(cs),profile:'fixture',engine:'chromium'});
                  if(removed.some(r=>r.status!=='removed')) throw Error('cookie deletion failed: '+JSON.stringify(removed));
                  const remaining = await c.cookiesInStore(api,store,false);
                  if(!remaining.some(x=>x.value==='root-fixture')) throw Error('same-name root cookie was removed');
                  const recovery=await c.restore(api,cs,{confirmation:'RESTORE 2',profile:'fixture',engine:'chromium'});
                  if(recovery.some(r=>r.status!=='restored')) throw Error('cookie restoration failed: '+JSON.stringify(recovery));
                  return {history:results,cookies:removed,restore:recovery,bookmarkPreserved:true};
                }''')
                page.locator('#history-access').click()
                page.locator('#domain').fill('repair-fixture.invalid')
                page.locator('#preview-history').click()
                page.wait_for_function("document.querySelectorAll('#rows tr').length === 1")
                assert page.locator('#delete').is_disabled()
                page.locator('#select-all').click()
                password = 'generated-ui-test-passphrase'
                page.locator('#new-password').fill(password)
                page.locator('#repeat-password').fill(password)
                with page.expect_download() as info:
                    page.locator('#backup').click()
                download = info.value
                backup_path = temp / 'fixture.otbrowser'
                download.save_as(backup_path)
                page.wait_for_function("!document.querySelector('#verify').disabled")
                assert page.locator('#new-password').input_value() == ''
                assert page.locator('#delete').is_disabled()
                page.locator('#verify-file').set_input_files(backup_path)
                page.locator('#verify-password').fill(password)
                page.locator('#verify').click()
                page.wait_for_function("!document.querySelector('#delete').disabled")
                page.locator('#delete-confirm').fill('DELETE 1')
                page.locator('#delete').click()
                page.wait_for_function("document.querySelector('#results').textContent.includes('removed')")
                assert page.locator('#delete').is_disabled()
                page.set_viewport_size({'width':390,'height':844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                assert not errors, errors
                print('Native Chromium smoke passed: hostname selection, bookmark/unrelated-history preservation, URL recovery,')
                print('same-name/different-path and partitioned cookie deletion/restoration, actual encrypted download/read-back,')
                print('confirmation-gated UI deletion, password clearing, 390px viewport, no JavaScript errors.')
                print(json.dumps(result))
            finally:
                context.close()

if __name__ == '__main__':
    main()
