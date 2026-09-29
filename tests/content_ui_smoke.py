"""Optional: Playwright + Chromium. Real CSV worker; mocked prompt/lyrics APIs, not Flask."""
import json
import mimetypes
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / '.omnitool-local/content-ui'
REPORTS.mkdir(parents=True, exist_ok=True)
env = Environment(loader=FileSystemLoader(ROOT/'templates'), autoescape=select_autoescape())

def url_for(name, **kw):
    return '/static/'+kw['filename'] if name == 'static' else '/'+name

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        pages = {'/prompts':'prompt_creator.html', '/lyrics':'lyrics_workbench.html', '/csv':'csv_editor.html'}
        if path in pages:
            data = env.get_template(pages[path]).render(signed_in=True, csrf_token='mock-csrf', url_for=url_for,
                get_flashed_messages=lambda **_:[], config={'kind':'kobold','url':'http://127.0.0.1:5001'},
                config_error='', system='Write a clear fixture prompt.').encode()
            mime='text/html; charset=utf-8'
        elif path.startswith('/static/') and Path(path).name in {p.name for p in (ROOT/'static').glob('*')}:
            file=ROOT/'static'/Path(path).name; data=file.read_bytes(); mime=mimetypes.guess_type(str(file))[0] or 'application/octet-stream'
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header('Content-Type',mime)
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'")
        self.end_headers(); self.wfile.write(data)
    def log_message(self,*_): pass

server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
base=f'http://127.0.0.1:{server.server_port}'
try:
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path=shutil.which('chromium') or None)
        context=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
        errors=[]
        jobs={};counter=[0]
        def mocked(route):
            request=route.request;path=urlsplit(request.url).path
            if path.endswith('/clear') or path.endswith('/stop'): result={'ok':True}
            elif '/tasks/' in path: result={'state':'succeeded','result':jobs[path.rsplit('/',1)[-1]],'error':''}
            else:
                counter[0]+=1;token=str(counter[0])
                if path.endswith('/status'): data={'model':'fixture-model','url':'http://127.0.0.1:5001'}
                elif path.endswith('/generate'): data={'content':'Fixture output <img src=x onerror="window.xss=1">','kind':'kobold','url':'http://127.0.0.1:5001'}
                elif path.endswith('/scan'):
                    data={'tracks':[{'index':0,'file':'Fixture.mp3','format':'MP3','artist':'Fixture Artist','title':'Fixture Track','album':'Fixture Album','duration':10,'has_lyrics':False,'inferred':False}], 'warnings':[],'warning_count':0}
                elif path.endswith('/prepare'):
                    data={'count':1,'output':'D:\\Tagged\\Album','rows':[{'file':'Fixture.mp3','status':'ready','reason':'','provider':'local-sidecar','plain':'Original test fixture lyrics','synced':''}]}
                elif path.endswith('/apply'): data={'count':1,'output':'D:\\Tagged\\Album'}
                else: raise AssertionError(path)
                jobs[token]=data; result={'task':token}
            route.fulfill(json=result)
        context.route('**/api/content/**',mocked)
        prompt=context.new_page();prompt.on('pageerror',lambda e:errors.append(str(e)));prompt.goto(base+'/prompts')
        prompt.locator('#prompt-idea').fill('A fixture subject');prompt.locator('#prompt-form button[type=submit]').click()
        expect(prompt.locator('#prompt-result')).to_have_value('Fixture output <img src=x onerror="window.xss=1">')
        assert prompt.evaluate('window.xss') is None
        prompt.screenshot(path=str(REPORTS/'prompt-desktop.png'),full_page=True)
        prompt.locator('#clear-content').click();expect(prompt.locator('#prompt-result')).to_have_value('')
        lyrics=context.new_page();lyrics.on('pageerror',lambda e:errors.append(str(e)));lyrics.goto(base+'/lyrics')
        lyrics.locator('#lyrics-root').fill('D:\\Music\\Album');lyrics.locator('#lyrics-scan-form button').click()
        expect(lyrics.locator('#lyrics-tracks tr')).to_have_count(1)
        assert not lyrics.locator('.track-choice').is_checked()
        lyrics.locator('.track-choice').check();lyrics.locator('#lyrics-out').fill('D:\\Tagged\\Album')
        lyrics.locator('#lyrics-prepare').click();expect(lyrics.locator('#lyrics-review')).to_be_visible()
        lyrics.locator('#lyrics-out').fill('D:\\Different');expect(lyrics.locator('#lyrics-review')).to_be_hidden()
        lyrics.locator('#lyrics-prepare').click();expect(lyrics.locator('#lyrics-review')).to_be_visible()
        lyrics.locator('#lyrics-confirm').fill('WRITE 1');lyrics.locator('#lyrics-apply').click()
        expect(lyrics.locator('#content-status')).to_contain_text('Created 1 tagged copies')
        csv=context.new_page();csv.on('pageerror',lambda e:errors.append(str(e)));csv.goto(base+'/csv')
        csv.locator('#csv-file').set_input_files({'name':'fixture.csv','mimeType':'text/csv','buffer':('item,value\n'+'\n'.join(f'ITEM-{i:03d},00{i}' for i in range(120))).encode()})
        csv.locator('#csv-import').click();expect(csv.locator('#csv-grid tbody tr')).to_have_count(50)
        csv.locator('#csv-next').click();expect(csv.locator('#csv-page')).to_have_text('2 / 3')
        csv.locator('#csv-filter').fill('ITEM-099');expect(csv.locator('#csv-grid tbody tr')).to_have_count(1)
        csv.locator('#csv-grid tbody .cell-button').first.click();csv.locator('#csv-edit-value').fill('edited')
        csv.locator('#csv-edit-form button[type=submit]').click();expect(csv.locator('#csv-grid tbody tr')).to_have_count(0)
        csv.locator('#csv-undo').click();expect(csv.locator('#csv-grid tbody tr')).to_have_count(1)
        csv.locator('#csv-filter').fill('');expect(csv.locator('#csv-grid tbody tr')).to_have_count(50)
        with csv.expect_download() as download: csv.locator('#csv-export').click()
        data=Path(download.value.path()).read_text();assert 'ITEM-119' in data and 'edited' not in data
        csv.screenshot(path=str(REPORTS/'csv-desktop.png'),full_page=True)
        for page,name in [(prompt,'prompt'),(lyrics,'lyrics'),(csv,'csv')]:
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),name
            page.screenshot(path=str(REPORTS/f'{name}-mobile.png'),full_page=True)
        assert not errors,errors
        browser.close()
        print('UI smoke passed: real CSV worker/edit/undo/filter/pagination/export; mocked prompt/lyrics flows; preview invalidation; 390px overflow; no JS errors.')
finally:
    server.shutdown();server.server_close();thread.join()
