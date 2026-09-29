"""Browser UI smoke with mocked extension APIs. Does NOT test native cookie/history APIs."""
import shutil
import tempfile
import threading
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
MOCK="""
window.chrome={runtime:{},permissions:{request:async()=>true},storage:{local:{get:async()=>({profileId:'ui-fixture'}),set:async()=>{}}},history:{
search:async()=>[{url:'https://example.test/a',title:'fixture'},{url:'https://other.test/?q=example.test',title:'unrelated'}],
getVisits:async({url})=>window.removed?[]:[{visitId:'1',visitTime:1,transition:'link',referringVisitId:'0'}],
deleteUrl:async()=>{window.removed=true;},addUrl:async()=>{}},cookies:{getAllCookieStores:async()=>[{id:'0'}]}};
"""

def main():
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(ROOT)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='omnitool-ui-fixture-') as directory, sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path=shutil.which('chromium') or None)
        try:
            page=browser.new_page(viewport={'width':1280,'height':1000},accept_downloads=True)
            page.add_init_script(MOCK)
            errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/extensions/browser-maintenance/workspace.html')
            page.locator('#history-access').click()
            page.locator('#domain').fill('example.test')
            page.locator('#preview-history').click()
            page.wait_for_function("document.querySelectorAll('#rows tr').length===1")
            assert page.locator('#delete').is_disabled()
            page.locator('#select-all').click()
            password='ui-fixture-generated-passphrase'
            page.locator('#new-password').fill(password);page.locator('#repeat-password').fill(password)
            with page.expect_download() as event:page.locator('#backup').click()
            path=Path(directory)/'fixture.otbrowser';event.value.save_as(path)
            page.wait_for_function("!document.querySelector('#verify').disabled")
            assert page.locator('#new-password').input_value()==''
            assert page.locator('#delete').is_disabled()
            page.locator('#verify-file').set_input_files(path)
            page.locator('#verify-password').fill(password);page.locator('#verify').click()
            page.wait_for_function("!document.querySelector('#delete').disabled")
            assert page.locator('#verify-password').input_value()==''
            page.locator('#delete-confirm').fill('wrong');page.locator('#delete').click()
            page.wait_for_function("document.querySelector('#message').textContent.includes('exactly')")
            assert not page.evaluate('Boolean(window.removed)')
            page.locator('#delete-confirm').fill('DELETE 1');page.locator('#delete').click()
            page.wait_for_function("document.querySelector('#results').textContent.includes('removed')")
            assert page.locator('#delete').is_disabled()
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            assert not errors,errors
            print('Mock-API UI smoke passed: exact-host preview, empty selection default, encrypted download/read-back,')
            print('passphrase clearing, wrong-confirmation rejection, single-use apply, 390px layout, no JS page errors.')
        finally:
            browser.close();server.shutdown();server.server_close()

if __name__=='__main__':main()
